import pygame
import random
import math
import time

# 定数の定義
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

GRAVITY = 0.5
BOUNCE_FACTOR = 0.7
FRICTION = 0.99
MIN_SPEED = 0.1
SPAWN_MAX_LEVEL = 10
LUCKY_CHANCE = 0.05
SUPER_LUCKY_CHANCE = 0.01
DROP_SPEED = 5
HEIGHT_BONUS_THRESHOLD = 300
FEVER_THRESHOLD = 10000
SPAWN_EVENT = pygame.USEREVENT + 1
SPAWN_INTERVAL = 1000  # ミリ秒単位の初期スポーン間隔

# 色の定義
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
BLACK = (0, 0, 0)

# フルーツの種類定義
FRUIT_TYPES = {
    1: {'name': 'グレープ', 'color': (128, 0, 128), 'radius': 20, 'score': 100},
    2: {'name': 'オレンジ', 'color': (255, 165, 0), 'radius': 25, 'score': 200},
    3: {'name': 'アップル', 'color': (255, 0, 0), 'radius': 30, 'score': 300},
    4: {'name': 'スイカ', 'color': (0, 255, 0), 'radius': 35, 'score': 500},
    5: {'name': 'メロン', 'color': (50, 205, 50), 'radius': 40, 'score': 800},
    6: {'name': 'パイナップル', 'color': (255, 223, 0), 'radius': 45, 'score': 1200},
    7: {'name': 'ドラゴンフルーツ', 'color': (255, 20, 147), 'radius': 50, 'score': 1800},
    8: {'name': 'マンゴー', 'color': (255, 140, 0), 'radius': 55, 'score': 2500},
    9: {'name': 'キング・フルーツ', 'color': (218, 165, 32), 'radius': 60, 'score': 3500},
    10: {'name': 'レジェンド・フルーツ', 'color': (255, 215, 0), 'radius': 65, 'score': 5000}
}

REWARDS = {
    'NORMAL': 1.0,
    'LUCKY': 3.0,
    'SUPER_LUCKY': 5.0,
    'CHAIN': 2.0,
    'FEVER': 3.0,
    'PERFECT': 10.0
}

# サウンドファイルのパス設定
COLLISION_SOUND_PATH = 'C:/Windows/Media/Windows Ding.wav'
BACKGROUND_MUSIC_PATH = 'C:/Windows/Media/Windows Notify.wav'
PERFECT_MERGE_SOUND_PATH = 'C:/Windows/Media/Windows Balloon.wav'

# サウンドの初期化
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()

# サウンドファイルのロード
COLLISION_SOUND = pygame.mixer.Sound(COLLISION_SOUND_PATH)
PERFECT_MERGE_SOUND = pygame.mixer.Sound(PERFECT_MERGE_SOUND_PATH)
BACKGROUND_MUSIC = BACKGROUND_MUSIC_PATH

# フルーツごとのサウンド設定
FRUIT_SOUNDS = {
    'グレープ': COLLISION_SOUND,
    'オレンジ': COLLISION_SOUND,
    'アップル': COLLISION_SOUND,
    'スイカ': COLLISION_SOUND,
    'メロン': COLLISION_SOUND,
    'パイナップル': COLLISION_SOUND,
    'ドラゴンフルーツ': COLLISION_SOUND,
    'マンゴー': COLLISION_SOUND,
    'キング・フルーツ': COLLISION_SOUND,
    'レジェンド・フルーツ': COLLISION_SOUND
}

# 背景音楽の再生
pygame.mixer.music.load(BACKGROUND_MUSIC)
pygame.mixer.music.play(-1)  # 無限ループ

# スコア関連
SCORE = 0
COMBO_COUNT = 0
FEVER_MODE = False

SCORE_FILE = 'scores.txt'

# フォントの設定
pygame.font.init()
FONT_LARGE = pygame.font.SysFont('Arial', 36)
FONT_MEDIUM = pygame.font.SysFont('Arial', 28)
FONT_SMALL = pygame.font.SysFont('Arial', 20)

# スプライトグループの作成
all_sprites = pygame.sprite.Group()
fruits_group = pygame.sprite.Group()
particles_group = pygame.sprite.Group()

def RAINBOW_COLOR():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def create_floating_text(text, x, y, color):
    floating_text = FloatingText(text, x, y, color)
    all_sprites.add(floating_text)

class FloatingText(pygame.sprite.Sprite):
    def __init__(self, text, x, y, color):
        super().__init__()
        self.image = FONT_SMALL.render(text, True, color)
        self.rect = self.image.get_rect(center=(x, y))
        self.velocity_y = -1
        self.lifetime = 60  # フレーム数

    def update(self):
        self.rect.y += self.velocity_y
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.kill()

class Fruit(pygame.sprite.Sprite):
    def __init__(self, x, y, level=None):
        super().__init__()
        if level is None:
            level = random.randint(1, min(4, SPAWN_MAX_LEVEL))
        self.level = level
        fruit_type = FRUIT_TYPES[self.level]
        self.radius = fruit_type['radius']
        self.color = fruit_type['color']
        self.name = fruit_type['name']
        
        # 物理演算パラメータ
        self.true_x = float(x)
        self.true_y = float(y)
        self.speed_y = 0  # 初速度を0に変更
        self.speed_x = 0
        self.merged = False
        self.stable = False
        self.stable_timer = 0
        
        self.create_image()
        self.rect = self.image.get_rect(center=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
        
        self.is_lucky = random.random() < LUCKY_CHANCE
        self.sparkle_effect = self.is_lucky
        self.bonus_multiplier = 2.0 if self.is_lucky else 1.0
        
        self.determine_fruit_type()
        self.chain_count = 0
        self.last_merge_time = time.time()

    def create_image(self):
        # フルーツの画像を作成
        size = self.radius * 2
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)

    def update(self):
        if self.merged:
            return
            
        # 物理演算の更新
        self.speed_y += GRAVITY
        
        # 位置の更新前に現在の位置を保存
        old_x = self.true_x
        old_y = self.true_y
        
        # 位置の更新
        self.true_x += self.speed_x
        self.true_y += self.speed_y
        
        # 画面端との衝突判定
        if self.true_x - self.radius < 0:
            self.true_x = self.radius
            self.speed_x = -self.speed_x * BOUNCE_FACTOR
        elif self.true_x + self.radius > SCREEN_WIDTH:
            self.true_x = SCREEN_WIDTH - self.radius
            self.speed_x = -self.speed_x * BOUNCE_FACTOR
            
        if self.true_y + self.radius > SCREEN_HEIGHT:
            self.true_y = SCREEN_HEIGHT - self.radius
            if abs(self.speed_y) > MIN_SPEED:
                self.speed_y = -self.speed_y * BOUNCE_FACTOR
            else:
                self.speed_y = 0
            self.speed_x *= FRICTION
        
        # フルーツ同士の衝突判定
        self.rect.centerx = int(self.true_x)
        self.rect.centery = int(self.true_y)
        
        collided = False
        for other in fruits_group:
            if other != self and not other.merged:
                dx = self.true_x - other.true_x
                dy = self.true_y - other.true_y
                distance = math.sqrt(dx * dx + dy * dy)
                min_dist = self.radius + other.radius
                
                if distance < min_dist:
                    collided = True
                    if self.can_merge(other):
                        self.merge_fruits(other)
                        break
                    
                    # 衝突応答
                    overlap = min_dist - distance
                    if distance > 0:
                        nx = dx / distance
                        ny = dy / distance
                        
                        # 位置の補正
                        push = overlap / 2
                        self.true_x += nx * push
                        self.true_y += ny * push
                        other.true_x -= nx * push
                        other.true_y -= ny * push
                        
                        # 速度の更新（反発）
                        relative_x = self.speed_x - other.speed_x
                        relative_y = self.speed_y - other.speed_y
                        
                        dot_product = relative_x * nx + relative_y * ny
                        
                        if dot_product > 0:
                            impulse = dot_product * BOUNCE_FACTOR
                            
                            self.speed_x -= impulse * nx
                            self.speed_y -= impulse * ny
                            other.speed_x += impulse * nx
                            other.speed_y += impulse * ny
                            
                            # 安定性の判定
                            if abs(self.speed_x) < MIN_SPEED and abs(self.speed_y) < MIN_SPEED:
                                self.stable = True
                                self.stable_timer = 60  # 60フレーム間安定

    def can_merge(self, other):
        """フルーツが合体可能かチェック"""
        if self.level != other.level:
            return False
        if self.merged or other.merged:
            return False
        return True

    def determine_fruit_type(self):
        """フルーツタイプの決定"""
        rand = random.random()
        if rand < SUPER_LUCKY_CHANCE:
            self.fruit_type = 'SUPER_LUCKY'
            self.bonus_multiplier = REWARDS['SUPER_LUCKY']
            self.color = RAINBOW_COLOR()
        elif rand < LUCKY_CHANCE:
            self.fruit_type = 'LUCKY'
            self.bonus_multiplier = REWARDS['LUCKY']
            self.color = GOLD
        else:
            self.fruit_type = 'NORMAL'
            self.bonus_multiplier = REWARDS['NORMAL']

    def is_perfect_merge(self, other):
        """完璧な合体判定"""
        height_bonus = self.true_y < HEIGHT_BONUS_THRESHOLD
        speed_bonus = abs(self.speed_y) > DROP_SPEED * 1.5
        return height_bonus and speed_bonus

    def calculate_merge_bonus(self, other):
        """合体ボーナスの計算"""
        bonus = self.bonus_multiplier * other.bonus_multiplier
        if self.chain_count > 0:
            bonus *= (1.0 + self.chain_count * 0.5)
        if FEVER_MODE:
            bonus *= REWARDS['FEVER']
        if self.is_perfect_merge(other):
            bonus *= REWARDS['PERFECT']
            self.show_perfect_effect()
        return bonus

    def show_perfect_effect(self):
        """完璧な合体時のエフェクト表示"""
        PERFECT_MERGE_SOUND.play()  # 完璧な合体時に別のサウンドを再生
        create_floating_text("Perfect!", self.rect.centerx, self.rect.centery, GOLD)
        show_special_effect(self.rect.centerx, self.rect.centery)

    def merge_fruits(self, other):
        """フルーツの合体処理"""
        score = calculate_merge_score(self, other)
        add_score(score)
        
        # サウンド再生
        FRUIT_SOUNDS[self.name].play()  # 各フルーツタイプに対応するサウンドを再生
        
        # 新しいフルーツを生成
        new_level = self.level + 1
        if new_level > SPAWN_MAX_LEVEL:
            return False
            
        new_fruit = Fruit((self.true_x + other.true_x) / 2, (self.true_y + other.true_y) / 2, new_level)
        fruits_group.add(new_fruit)
        all_sprites.add(new_fruit)
        
        # エフェクトの作成
        create_merge_effect(new_fruit.true_x, new_fruit.true_y, new_fruit.color)
        
        # 元のフルーツを削除
        self.merged = True
        other.merged = True
        self.kill()
        other.kill()
        
        return True

def calculate_merge_score(fruit1, fruit2):
    """フルーツ合体時のスコア計算"""
    base_score = FRUIT_TYPES[fruit1.level]['score']
    bonus = fruit1.calculate_merge_bonus(fruit2)
    return int(base_score * bonus)

def add_score(points):
    """スコアを加算"""
    global SCORE, COMBO_COUNT, FEVER_MODE
    SCORE += points
    COMBO_COUNT += 1
    create_floating_text(f"+{points}", 750, 50, WHITE)
    
    # フィーバーモードの判定
    if SCORE >= FEVER_THRESHOLD and not FEVER_MODE:
        FEVER_MODE = True
        create_floating_text("Fever Mode!", SCREEN_WIDTH//2, SCREEN_HEIGHT//2, GOLD)

def create_merge_effect(x, y, color):
    """合体エフェクトを作成"""
    for _ in range(20):
        particle = Particle(x, y, color)
        particles_group.add(particle)

def play_merge_sound(level):
    """フルーツの合体時にサウンドを再生"""
    try:
        if level <= 3:
            FRUIT_SOUNDS['grape'].play()
        elif level <= 6:
            FRUIT_SOUNDS['orange'].play()
        elif level <= 9:
            FRUIT_SOUNDS['apple'].play()
        else:
            FRUIT_SOUNDS['watermelon'].play()
    except Exception as e:
        print("サウンド再生エラー:", e)

def create_fruit(x, y, level=None):
    """フルーツを生成する関数"""
    if level is None:
        level = random.randint(1, SPAWN_MAX_LEVEL)
    
    fruit = Fruit(x, y, level)
    fruit.merged = False  # 合体フラグを初期化
    
    # スプライトグループに追加
    all_sprites.add(fruit)
    fruits_group.add(fruit)
    
    return fruit

def draw_score(screen):
    score_text = FONT_LARGE.render(f"Score: {SCORE}", True, WHITE)
    screen.blit(score_text, (20, 20))

def check_game_over():
    for fruit in fruits_group:
        # フルーツが画面上部に到達し、かつ一定時間停止している場合はゲームオーバー
        if fruit.rect.top <= 100 and abs(fruit.speed_y) < 0.1:
            fruit.stable_timer += 1
            if fruit.stable_timer > FPS * 3:  # 3秒上部で停止したらゲームオーバー
                return True
        else:
            fruit.stable_timer = 0
    return False

def save_score(score):
    with open(SCORE_FILE, 'a') as f:
        f.write(f'{score}\n')

def load_scores():
    try:
        with open(SCORE_FILE, 'r') as f:
            scores = [int(line.strip()) for line in f]
        return sorted(scores, reverse=True)[:5]
    except FileNotFoundError:
        return []

def show_ranking(surface):
    scores = load_scores()
    surface.fill(BLACK)
    title = FONT_MEDIUM.render('Ranking', True, WHITE)
    surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))

    for idx, score in enumerate(scores):
        rank_text = FONT_SMALL.render(f'{idx + 1}. {score}', True, WHITE)
        surface.blit(rank_text, (SCREEN_WIDTH // 2 - rank_text.get_width() // 2, 150 + idx * 40))

    pygame.display.flip()
    pygame.time.wait(5000)

def show_game_over(surface):
    surface.fill(BLACK)
    game_over_text = FONT_LARGE.render('GAME OVER', True, WHITE)
    score_text = FONT_MEDIUM.render(f'Score: {SCORE}', True, WHITE)
    restart_text = FONT_SMALL.render('Press SPACE to restart', True, WHITE)
    
    surface.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//3))
    surface.blit(score_text, (SCREEN_WIDTH//2 - score_text.get_width()//2, SCREEN_HEIGHT//2))
    surface.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT - 100))

    pygame.display.flip()
    pygame.time.wait(3000)

def play_merge_animation(x, y, color):
    """拡大縮小アニメーション"""
    for i in range(1, 11):
        temp_radius = 10 * i
        pygame.draw.circle(screen, color, (int(x), int(y)), temp_radius, 2)
        pygame.display.flip()
        pygame.time.wait(20)
        # フレームごとに透明度を下げる
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (int(x), int(y)), temp_radius, 2)
        screen.blit(s, (0, 0))
        pygame.display.flip()

def show_special_effect(x, y):
    """強化されたパーティクルエフェクト"""
    for _ in range(50):
        Particle(x, y, GOLD).add(particles_group)

def increase_difficulty():
    """難易度を増加させる関数"""
    global GRAVITY, SPAWN_INTERVAL
    GRAVITY += 0.05
    SPAWN_INTERVAL = max(500, SPAWN_INTERVAL - 100)
    pygame.time.set_timer(SPAWN_EVENT, SPAWN_INTERVAL)

def settings_menu():
    """設定メニューの関数（未実装）"""
    pass

class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.size = random.randint(2, 4)
        self.image = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (self.size, self.size), self.size)
        self.rect = self.image.get_rect(center=(x, y))
        self.velocity = [random.uniform(-2, 2), random.uniform(-2, -5)]
        self.lifetime = random.randint(20, 40)
        self.alpha = 255

    def update(self):
        # 位置の更新
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
        self.velocity[1] += 0.2  # 重力
        
        # 透明度の更新
        self.alpha -= 255 / self.lifetime
        if self.alpha <= 0:
            self.kill()
        else:
            self.image.set_alpha(self.alpha)

class Game:
    def __init__(self):
        self.next_fruit = None
        self.can_drop = True
        self.drop_cooldown = 500  # ミリ秒
        self.last_drop_time = 0
        self.preview_x = SCREEN_WIDTH // 2
        self.game_over = False
        self.score = 0
        
        # 次のフルーツを準備
        self.prepare_next_fruit()

    def prepare_next_fruit(self):
        """次のフルーツを準備する"""
        self.next_fruit = random.randint(1, min(4, SPAWN_MAX_LEVEL))  # 最初は小さいフルーツのみ
        self.can_drop = True

    def update(self, mouse_x):
        """ゲーム状態の更新"""
        # プレビュー位置の更新
        self.preview_x = max(50, min(mouse_x, SCREEN_WIDTH - 50))
        
        # ドロップクールダウンの更新
        current_time = pygame.time.get_ticks()
        if not self.can_drop and current_time - self.last_drop_time >= self.drop_cooldown:
            self.can_drop = True

    def draw_preview(self, screen):
        """次のフルーツのプレビューを描画"""
        if self.next_fruit and self.can_drop:
            preview_color = FRUIT_TYPES[self.next_fruit]['color']
            preview_radius = FRUIT_TYPES[self.next_fruit]['radius']
            pygame.draw.circle(screen, preview_color, (self.preview_x, 50), preview_radius, 2)
            # ドロップ可能ラインの描画
            pygame.draw.line(screen, WHITE, (self.preview_x, 0), (self.preview_x, 100), 1)

def main():
    global screen, SCORE
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("スイカゲーム")
    clock = pygame.time.Clock()

    game = Game()
    running = True

    while running:
        clock.tick(FPS)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左クリック
                if game.can_drop and not game.game_over:
                    # フルーツをドロップ
                    new_fruit = Fruit(game.preview_x, 50, game.next_fruit)
                    fruits_group.add(new_fruit)
                    all_sprites.add(new_fruit)
                    
                    # クールダウンの設定
                    game.can_drop = False
                    game.last_drop_time = pygame.time.get_ticks()
                    
                    # 次のフルーツを準備
                    game.prepare_next_fruit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game.game_over:
                    # ゲームリセット
                    game = Game()
                    SCORE = 0
                    all_sprites.empty()
                    fruits_group.empty()
                    particles_group.empty()

        # ゲーム状態の更新
        if not game.game_over:
            game.update(mouse_x)
            all_sprites.update()
            particles_group.update()

            # ゲームオーバー判定
            if check_game_over():
                game.game_over = True
                save_score(SCORE)

        # 描画
        screen.fill(BLACK)
        
        # プレビューの描画
        if not game.game_over:
            game.draw_preview(screen)
        
        all_sprites.draw(screen)
        draw_score(screen)
        
        if game.game_over:
            show_game_over(screen)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
