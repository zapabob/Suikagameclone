# -*- coding: utf-8 -*-
import pygame
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import numpy as np
import numpy

def create_tone(frequency, duration, amplitude=4096):
    sample_rate = 44100
    t = numpy.linspace(0, duration, int(sample_rate * duration))
    wave = amplitude * numpy.sin(2.0 * numpy.pi * frequency * t)
    return wave.astype(numpy.int16)

def create_game_bgm():
    # メロディーの音符（C major scale: C4, D4, E4, F4, G4, A4, B4, C5）
    notes = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
    }
    
    # メロディーパターン
    melody = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4'] * 2
    duration = 0.25  # 音の長さ（秒）
    
    # メロディーを生成
    wave = numpy.array([], dtype=numpy.int16)
    for note in melody:
        tone = create_tone(notes[note], duration)
        wave = numpy.concatenate([wave, tone])
    
    # ベース音を追加
    bass_freq = notes['C4'] / 2  # 1オクターブ下
    bass = create_tone(bass_freq, duration * len(melody), amplitude=2048)
    
    # メロディーとベースを合成
    combined = wave + numpy.resize(bass, wave.shape)
    return pygame.sndarray.make_sound(combined)

# 初期化
pygame.init()
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
TOTAL_WIDTH = SCREEN_WIDTH * 2 + 100  # 2画面 + 中央の余白
screen = pygame.display.set_mode((TOTAL_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("スイカゲーム AI vs Human")

# 画面の定義
LEFT_SCREEN = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
RIGHT_SCREEN = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

# フルーツの定義
FRUIT_NAMES = {
    1: "さくらんぼ",
    2: "ぶどう",
    3: "みかん",
    4: "柿",
    5: "りんご",
    6: "梨",
    7: "もも",
    8: "パイナップル",
    9: "メロン",
    10: "すいか",
    11: "金のすいか"
}

COLORS = {
    1: (255, 0, 0),      # さくらんぼ：赤
    2: (128, 0, 128),    # ぶどう：紫
    3: (255, 165, 0),    # みかん：オレンジ
    4: (255, 140, 0),    # 柿：オレンジ
    5: (255, 69, 0),     # りんご：赤
    6: (255, 215, 0),    # 梨：黄色
    7: (255, 192, 203),  # もも：ピンク
    8: (255, 223, 0),    # パイナップル：黄色
    9: (50, 205, 50),    # メロン：緑
    10: (0, 255, 0),     # すいか：緑
    11: (255, 215, 0)    # 金のすいか：金
}

class Fruit:
    def __init__(self, x, level):
        self.x = x
        self.y = 0
        self.level = level
        self.radius = 15 + (level * 5)
        self.vel_x = 0
        self.vel_y = 0
        self.fixed = False
        self.gravity = 0.5
        self.restitution = 0.3  # 反発係数
        self.friction = 0.8     # 摩擦係数
        self.angular_vel = 0    # 回転速度
        self.rotation = 0       # 回転角度
        self.z = 0             # 奥行き座標

    def update(self):
        if not self.fixed:
            # 重力と速度の更新
            self.vel_y += self.gravity
            self.x += self.vel_x
            self.y += self.vel_y
            
            # 回転の更新
            self.rotation += self.angular_vel
            self.angular_vel *= 0.95  # 回転の減衰
            
            # 壁との衝突
            if self.x < self.radius:
                self.x = self.radius
                self.vel_x = -self.vel_x * self.restitution
                self.angular_vel += self.vel_x * 0.1  # 衝突による回転
            elif self.x > SCREEN_WIDTH - self.radius:
                self.x = SCREEN_WIDTH - self.radius
                self.vel_x = -self.vel_x * self.restitution
                self.angular_vel -= self.vel_x * 0.1  # 衝突による回転
            
            # 床との衝突
            if self.y > SCREEN_HEIGHT - 20 - self.radius:
                self.y = SCREEN_HEIGHT - 20 - self.radius
                if abs(self.vel_y) < 0.5 and abs(self.vel_x) < 0.5:
                    self.fixed = True
                    self.vel_x = 0
                    self.vel_y = 0
                    self.angular_vel = 0
                else:
                    self.vel_y = -self.vel_y * self.restitution
                    self.vel_x *= self.friction
                    self.angular_vel *= 0.8

    def check_collision(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.sqrt(dx * dx + dy * dy)
        min_distance = self.radius + other.radius
        
        if distance < min_distance:
            # 衝突応答
            if distance == 0:
                angle = random.uniform(0, 2 * math.pi)
                dx = math.cos(angle)
                dy = math.sin(angle)
                distance = 0.1
            else:
                dx = dx / distance
                dy = dy / distance
            
            # めり込み解消
            overlap = (min_distance - distance) * 0.5
            if not self.fixed:
                self.x += dx * overlap
                self.y += dy * overlap
                self.z += random.uniform(-0.1, 0.1)  # 奥行きの微調整
            if not other.fixed:
                other.x -= dx * overlap
                other.y -= dy * overlap
                other.z -= random.uniform(-0.1, 0.1)  # 奥行きの微調整
            
            # 速度の更新
            if not (self.fixed and other.fixed):
                # 相対速度
                rel_vel_x = self.vel_x - other.vel_x
                rel_vel_y = self.vel_y - other.vel_y
                
                # 衝突による速度変化
                normal_vel = rel_vel_x * dx + rel_vel_y * dy
                impulse = -(1 + self.restitution) * normal_vel
                
                if not self.fixed:
                    self.vel_x += impulse * dx
                    self.vel_y += impulse * dy
                    self.angular_vel += (dx * self.vel_y - dy * self.vel_x) * 0.1
                if not other.fixed:
                    other.vel_x -= impulse * dx
                    other.vel_y -= impulse * dy
                    other.angular_vel -= (dx * other.vel_y - dy * other.vel_x) * 0.1
                
                # 摩擦の適用
                self.vel_x *= self.friction
                self.vel_y *= self.friction
                other.vel_x *= other.friction
                other.vel_y *= other.friction
            
            return True
        return False

    def draw(self, screen):
        # 影の描画
        shadow_y = SCREEN_HEIGHT - 20
        shadow_scale = 0.3 - (self.y / SCREEN_HEIGHT) * 0.2  # 高さに応じて影のサイズを変更
        shadow_alpha = int(255 * (1 - (self.y / SCREEN_HEIGHT) * 0.7))  # 高さに応じて影の透明度を変更
        shadow_surface = pygame.Surface((self.radius * 2, self.radius * shadow_scale), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, shadow_alpha), 
                          shadow_surface.get_rect())
        screen.blit(shadow_surface, 
                   (self.x - self.radius,
                    shadow_y - self.radius * shadow_scale / 2))

        # フルーツの描画（回転を考慮）
        fruit_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(fruit_surface, COLORS[self.level], 
                         (self.radius, self.radius), 
                         self.radius)
        
        # 光沢効果
        highlight_radius = self.radius * 0.3
        highlight_pos = (int(self.radius - self.radius * 0.3),
                        int(self.radius - self.radius * 0.3))
        pygame.draw.circle(fruit_surface, (255, 255, 255, 128),
                         highlight_pos, int(highlight_radius))
        
        # 回転を適用
        rotated_surface = pygame.transform.rotate(fruit_surface, self.rotation)
        screen.blit(rotated_surface, 
                   (self.x - rotated_surface.get_width()//2,
                    self.y - rotated_surface.get_height()//2))

# AIの深層学習モデル
class SuikaNet(nn.Module):
    def __init__(self):
        super(SuikaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 30, 512)
        self.fc2 = nn.Linear(512, SCREEN_WIDTH)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 20 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SuikaAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = SuikaNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.stats = {
            'games_played': 0,
            'max_score': 0,
            'total_merges': 0,
            'losses': []
        }

    def get_state(self, fruits, current_level):
        # ゲーム状態を2D配列に変換
        state = np.zeros((SCREEN_HEIGHT//10, SCREEN_WIDTH//10))
        for fruit in fruits:
            x = int(fruit.x / 10)
            y = int(fruit.y / 10)
            if 0 <= x < SCREEN_WIDTH//10 and 0 <= y < SCREEN_HEIGHT//10:
                state[y, x] = fruit.level / 11.0
        
        # PyTorchテンソルに変換
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        return state_tensor.to(self.device)

    def select_position(self, fruits, current_level):
        state = self.get_state(fruits, current_level)
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                action_values = self.model(state)
                return action_values.max(1)[1].item()
        else:
            return random.randint(0, SCREEN_WIDTH-1)

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([s[0] for s in batch])
        actions = torch.tensor([s[1] for s in batch], device=self.device)
        rewards = torch.tensor([s[2] for s in batch], device=self.device, dtype=torch.float32)
        next_states = torch.cat([s[3] for s in batch])
        dones = torch.tensor([s[4] for s in batch], device=self.device, dtype=torch.float32)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.stats['losses'].append(loss.item())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path="suika_model.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': self.stats
        }, path)

    def load_model(self, path="suika_model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.stats = checkpoint['stats']
            print("モデルを読み込みました")
        else:
            print("新規モデルで開始します")

# コンボシステムとエフェクトの追加
class ComboSystem:
    def __init__(self):
        self.combo = 0
        self.combo_timer = 0
        self.max_combo = 0
        self.total_score = 0
        
    def add_combo(self):
        self.combo += 1
        self.combo_timer = 120  # 2秒
        self.max_combo = max(self.max_combo, self.combo)
        
    def update(self):
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo = 0

class ScoreEffect:
    def __init__(self, x, y, text, color):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.life = 60
        self.vel_y = -2
        
    def update(self):
        self.y += self.vel_y
        self.life -= 1
        return self.life > 0
        
    def draw(self, screen):
        alpha = min(255, self.life * 4)
        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, self.color)
        text_surface.set_alpha(alpha)
        screen.blit(text_surface, (self.x, self.y))

def draw_battle_info(screen, left_score, right_score, time_left, left_combo, right_combo):
    font = pygame.font.Font(None, 36)
    
    # スコア表示
    left_text = font.render(f"AI: {left_score}", True, (0, 0, 0))
    right_text = font.render(f"プレイヤー: {right_score}", True, (0, 0, 0))
    vs_text = font.render("VS", True, (255, 0, 0))
    time_text = font.render(f"残り時間: {time_left//60}秒", True, (0, 0, 0))
    
    # コンボ表示
    left_combo_text = font.render(f"コンボ: {left_combo}!", True, (255, 0, 0)) if left_combo > 1 else None
    right_combo_text = font.render(f"コンボ: {right_combo}!", True, (255, 0, 0)) if right_combo > 1 else None
    
    # 中央部分に表示
    screen.blit(left_text, (SCREEN_WIDTH + 20, 50))
    screen.blit(vs_text, (SCREEN_WIDTH + 40, SCREEN_HEIGHT//2))
    screen.blit(right_text, (SCREEN_WIDTH + 20, 100))
    screen.blit(time_text, (SCREEN_WIDTH + 20, 150))
    
    if left_combo_text:
        screen.blit(left_combo_text, (20, 50))
    if right_combo_text:
        screen.blit(right_combo_text, (SCREEN_WIDTH + 120, 50))

def draw_fruits(screen, fruits):
    # フルーツを奥行きでソート（z座標が小さい順）
    sorted_fruits = sorted(fruits, key=lambda f: f.z)
    for fruit in sorted_fruits:
        fruit.draw(screen)

def main():
    # 左画面（AI）の状態
    left_fruits = []
    left_current_fruit = None
    left_next_level = 1
    left_score = 0
    left_game_over = False
    left_combo = ComboSystem()
    left_effects = []
    
    # 右画面（人間）の状態
    right_fruits = []
    right_current_fruit = None
    right_next_level = 1
    right_score = 0
    right_game_over = False
    right_combo = ComboSystem()
    right_effects = []
    
    # 共通の状態
    running = True
    clock = pygame.time.Clock()
    battle_time = 180 * 60  # 3分間
    
    # AIの初期化
    ai = SuikaAI()
    ai.load_model()
    
    # BGMの初期化
    try:
        pygame.mixer.init()
        game_bgm = create_game_bgm()
        game_bgm.play(-1)  # 無限ループ再生
    except Exception as e:
        print(f"BGMの初期化��失敗しまし��: {e}")
    
    while running and battle_time > 0:
        LEFT_SCREEN.fill((255, 255, 255))
        RIGHT_SCREEN.fill((255, 255, 255))
        screen.fill((240, 240, 240))
        
        battle_time -= 1
        left_combo.update()
        right_combo.update()
        
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION and right_current_fruit and not right_current_fruit.fixed:
                mouse_x = event.pos[0] - (SCREEN_WIDTH + 100)
                if 0 <= mouse_x <= SCREEN_WIDTH:
                    right_current_fruit.x = max(min(mouse_x, SCREEN_WIDTH - right_current_fruit.radius), 
                                           right_current_fruit.radius)
            elif event.type == pygame.MOUSEBUTTONDOWN and right_current_fruit and not right_current_fruit.fixed:
                right_current_fruit.vel_y = 5

        # AI（左画面）の更新
        if left_current_fruit is None and not left_game_over:
            state = ai.get_state(left_fruits, left_next_level)
            action = ai.select_position(left_fruits, left_next_level)
            x_pos = action
            left_current_fruit = Fruit(x_pos, left_next_level)
            left_current_fruit.vel_y = 5
            left_next_level = random.randint(1, 5)

        # 人間（右画面）の更新
        if right_current_fruit is None and not right_game_over:
            mouse_x = pygame.mouse.get_pos()[0] - (SCREEN_WIDTH + 100)
            right_current_fruit = Fruit(mouse_x, right_next_level)
            right_next_level = random.randint(1, 5)

        # 左画面の更新
        if left_current_fruit:
            left_current_fruit.update()
            if left_current_fruit.fixed:
                left_fruits.append(left_current_fruit)
                left_current_fruit = None

        # 右画面の更新
        if right_current_fruit:
            right_current_fruit.update()
            if right_current_fruit.fixed:
                right_fruits.append(right_current_fruit)
                right_current_fruit = None

        # 衝突判定と合体（左画面）
        merge_occurred_left = False
        merge_occurred_right = False

        for i, fruit in enumerate(left_fruits):
            fruit.update()
            for other in left_fruits[i+1:]:
                if fruit.check_collision(other):
                    if fruit.level == other.level and fruit.level < 11:
                        new_fruit = Fruit((fruit.x + other.x)/2, fruit.level + 1)
                        new_fruit.y = (fruit.y + other.y)/2
                        left_fruits.remove(fruit)
                        left_fruits.remove(other)
                        left_fruits.append(new_fruit)
                        left_score += (fruit.level + 1) * 10
                        merge_occurred_left = True
                        merge_x, merge_y = new_fruit.x, new_fruit.y
                        break

        # 衝突判定と合体（右画面）
        for i, fruit in enumerate(right_fruits):
            fruit.update()
            for other in right_fruits[i+1:]:
                if fruit.check_collision(other):
                    if fruit.level == other.level and fruit.level < 11:
                        new_fruit = Fruit((fruit.x + other.x)/2, fruit.level + 1)
                        new_fruit.y = (fruit.y + other.y)/2
                        right_fruits.remove(fruit)
                        right_fruits.remove(other)
                        right_fruits.append(new_fruit)
                        right_score += (fruit.level + 1) * 10
                        merge_occurred_right = True
                        merge_x, merge_y = new_fruit.x, new_fruit.y
                        break

        # ゲームオーバー判定
        for fruit in left_fruits:
            if fruit.fixed and fruit.y < 100:
                left_game_over = True
        for fruit in right_fruits:
            if fruit.fixed and fruit.y < 100:
                right_game_over = True

        # マージ時のコンボとエフェクト
        if merge_occurred_left:
            left_combo.add_combo()
            bonus = left_combo.combo * 100
            left_score += bonus
            left_effects.append(ScoreEffect(merge_x, merge_y, f"+{bonus}", (255, 0, 0)))

        if merge_occurred_right:
            right_combo.add_combo()
            bonus = right_combo.combo * 100
            right_score += bonus
            right_effects.append(ScoreEffect(merge_x, merge_y, f"+{bonus}", (255, 0, 0)))

        # エフェクトの更新と描画
        left_effects = [effect for effect in left_effects if effect.update()]
        right_effects = [effect for effect in right_effects if effect.update()]
        
        for effect in left_effects:
            effect.draw(LEFT_SCREEN)
        for effect in right_effects:
            effect.draw(RIGHT_SCREEN)

        # 描画
        # 左画面の描画
        pygame.draw.line(LEFT_SCREEN, (0, 0, 0), (0, SCREEN_HEIGHT - 20), 
                        (SCREEN_WIDTH, SCREEN_HEIGHT - 20), 2)
        draw_fruits(LEFT_SCREEN, left_fruits)
        if left_current_fruit:
            left_current_fruit.draw(LEFT_SCREEN)

        # 右画面の描画
        pygame.draw.line(RIGHT_SCREEN, (0, 0, 0), (0, SCREEN_HEIGHT - 20), 
                        (SCREEN_WIDTH, SCREEN_HEIGHT - 20), 2)
        draw_fruits(RIGHT_SCREEN, right_fruits)
        if right_current_fruit:
            right_current_fruit.draw(RIGHT_SCREEN)

        # メイン画面に両方の画面を描画
        screen.blit(LEFT_SCREEN, (0, 0))
        screen.blit(RIGHT_SCREEN, (SCREEN_WIDTH + 100, 0))
        
        # 対戦情報の表示
        draw_battle_info(screen, left_score, right_score, battle_time, 
                        left_combo.combo, right_combo.combo)

        # ゲーム終了判定と結果表示
        if battle_time <= 0 or (left_game_over and right_game_over):
            font = pygame.font.Font(None, 72)
            if left_score > right_score:
                result_text = "AI の勝利!"
                result_color = (255, 0, 0)
            elif right_score > left_score:
                result_text = "プレイヤーの勝利!"
                result_color = (0, 255, 0)
            else:
                result_text = "引き分け!"
                result_color = (0, 0, 255)
            
            # 結果表示
            text = font.render(result_text, True, result_color)
            screen.blit(text, (TOTAL_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2))
            
            # 戦績表示
            stats_font = pygame.font.Font(None, 36)
            stats_texts = [
                f"AIの最大コンボ: {left_combo.max_combo}",
                f"プレイヤーの最大コンボ: {right_combo.max_combo}",
                f"AIのスコア: {left_score}",
                f"プレイヤーのスコア: {right_score}"
            ]
            
            for i, text in enumerate(stats_texts):
                surface = stats_font.render(text, True, (0, 0, 0))
                screen.blit(surface, (TOTAL_WIDTH//2 - surface.get_width()//2, 
                                    SCREEN_HEIGHT//2 + 50 + i * 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()