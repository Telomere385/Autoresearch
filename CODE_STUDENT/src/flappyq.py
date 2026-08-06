from itertools import cycle
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from sys import argv

import pygame
from pygame.locals import *


FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'


def asset_path(relative_path):
    return str(BASE_DIR / relative_path)


def data_path(*parts):
    return str(DATA_DIR.joinpath(*parts))


def input_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    q_table_path = DATA_DIR / 'q_tables' / path
    if q_table_path.exists():
        return str(q_table_path)
    return str(BASE_DIR / path)


def output_path(*parts):
    if RUN_DIR is not None:
        return str(RUN_DIR.joinpath(*parts))
    return data_path(*parts)

PIPEGAPSIZE  = 100 # gap between upper and lower pipe
PIPEWIDTH = 52
BIRDWIDTH = 34
BIRDHEIGHT = 24
BIRDDIAMETER = np.sqrt(BIRDHEIGHT**2 + BIRDWIDTH**2) # the bird rotates in the game, so we use it's maximum extent
SKY = 0 # location of sky
GROUND = (512*0.79)-1 # location of ground
PLAYERX = 57 # location of bird

reward = 0
olddx = 0
olddy = 0
oldvy = 0
oldflap = 0

counter = 0
rewsum = 0

highscore = 0
totscore = 0

screen = True 
episode_rewards = []  # 用来存每一局的总reward #Plot
ave_rewards = [] #Plot
Q = None
MAX_EPISODES = 1000
SAVE_EVERY = 100
RUN_DIR = None
LEARNING_ENABLED = True
SAMPLE_T = 3
GAMMA = 0.95
EPSILON = 0.001
EPSILON_START = 0.001
EPSILON_END = 0.001
EPSILON_DECAY = 1.0
LEARNING_RATE_INITIAL = 0.6
LEARNING_RATE_MIN = 0.1
LEARNING_RATE_DECAY = 0.9995
DISCRETIZATION_R = 25
DISCRETIZATION_RV = 2
PIPE_PASSED_REWARD = 10
DID_NOT_DIE_REWARD = 0.05
DIE_REWARD = -50
PLAYER_FLAP_ACC = -10
EPISODE_SCORES = []
PROGRESS_METRICS = []
ACTIVE_CONFIG = None


class SilentSound:
    def play(self):
        pass


def reset_run_state():
    global reward, olddx, olddy, oldvy, oldflap
    global counter, rewsum, highscore, totscore
    global episode_rewards, ave_rewards, EPISODE_SCORES, PROGRESS_METRICS

    reward = 0
    olddx = 0
    olddy = 0
    oldvy = 0
    oldflap = 0
    counter = 0
    rewsum = 0
    highscore = 0
    totscore = 0
    episode_rewards = []
    ave_rewards = []
    EPISODE_SCORES = []
    PROGRESS_METRICS = []


# 3.3.2 huristic initialisation of Q values
def prefill_Q_heuristic(Q, r=25, rv=2):
    """
    Heuristic warm-start for Q(dx, dy, vy, a).
    - dx: distance to next pipe center (pixels), binned by r
    - dy: vertical distance to pipe gap center (pixels), binned by r, with +512 shift in your code
    - vy: vertical velocity, binned by rv, with +50 shift in your code
    a=0: no flap, a=1: flap
    """

    dx_bins, dy_bins, vy_bins, nA = Q.shape
    assert nA == 2

    # Reconstruct approximate (dx, dy, vy) values at bin centers
    dx_vals = (np.arange(dx_bins) + 0.5) * r                 # [0, ...]
    dy_vals = (np.arange(dy_bins) + 0.5) * r - 512           # reverse your (dy+512)//r
    vy_vals = (np.arange(vy_bins) + 0.5) * rv - 50           # reverse your (vy+50)//rv

    # Build 3D grids
    DX = dx_vals[:, None, None]   # shape (dx_bins,1,1)
    DY = dy_vals[None, :, None]   # shape (1,dy_bins,1)
    VY = vy_vals[None, None, :]   # shape (1,1,vy_bins)

    # --- Heuristic "urgency" to flap ---
    # Bigger means flap is more desirable.
    # Intuition:
    #   DY > 0 => bird is BELOW the gap center (needs to go up) -> flap more
    #   VY > 0 => moving downward -> flap more
    #   DX small => pipe is close -> decisions matter more

    close = np.exp(-DX / 60.0)

    # 只在明显需要上升时才增加flap偏好：DY 大于某个阈值才开始起作用
    dy_thresh = 50.0
    need_up = np.maximum(DY - dy_thresh, 0.0)  # ReLU: DY<=50时为0

    # 下坠时更需要 flap，但上升时就别鼓励 flap
    need_flap_from_vy = np.maximum(VY, 0.0)

    urgency = close * (0.004 * need_up + 0.03 * need_flap_from_vy)

    # 给 flap 一个小惩罚（让 noflap 更容易赢一些）
    flap_penalty = 0.02 * close
    # flap_penalty = 0.05 * close

    base = -0.1 * close
    scale = 0.2

    Q[..., 0] = scale * (base - urgency)          # no flap
    Q[..., 1] = scale * (base + urgency - flap_penalty)  # flap
    # Convert to initial Q preferences
    # Make both actions slightly negative far away (so values not all zero),
    # then bias flap vs no-flap by urgency.

    # Optional: clamp to avoid crazy magnitudes
    np.clip(Q, -5.0, 5.0, out=Q)

    return Q

def prefill_Q_simple(Q, r=25, rv=2):
    dx_bins, dy_bins, vy_bins, nA = Q.shape
    assert nA == 2

    # 近似恢复物理量
    dx_vals = (np.arange(dx_bins) + 0.5) * r
    dy_vals = (np.arange(dy_bins) + 0.5) * r - 512
    vy_vals = (np.arange(vy_bins) + 0.5) * rv - 50

    DX = dx_vals[:, None, None]
    DY = dy_vals[None, :, None]
    VY = vy_vals[None, None, :]

    # 距离权重：越近越重要
    close = 1.0 / (1.0 + DX / 80.0)

    # 核心逻辑：
    # DY>0 (在缺口下方) + VY>0 (向下掉) => 倾向 flap
    preference = close * (0.01 * DY + 0.05 * VY)

    Q[..., 0] = -preference   # no flap
    Q[..., 1] =  preference   # flap

    np.clip(Q, -5.0, 5.0, out=Q)
    return Q

def prefill_Q_gap_center_bins(Q, r=25, rv=2,
                              k_dybin=0.20, k_vybin=0.10, k_near=0.15,
                              near_dx_bins=3, clip=1.0):
    dx_bins, dy_bins, vy_bins, nA = Q.shape
    assert nA == 2

    dy0 = 512 // r
    vy0 = 50 // rv

    Dx = np.arange(dx_bins)[:, None, None]
    Dy = np.arange(dy_bins)[None, :, None]
    Vy = np.arange(vy_bins)[None, None, :]

    # 关键：+ 0*Dx 让 A 直接拥有 dx 维度 -> (dx,dy,vy)
    A = k_dybin * (Dy - dy0) + k_vybin * (Vy - vy0) + 0.0 * Dx
    A -= k_near * (Dx < near_dx_bins).astype(float)
    A = np.clip(A, -clip, clip)

    Q[..., 0] = -0.5 * A
    Q[..., 1] = +0.5 * A
    return Q


if(len(argv)) == 2:
    if argv[1] == 'play':
        Q = np.load(data_path('q_tables', 'Q_last_lr_0.6.npy'))
        FPS = 30
        screen = True
    elif argv[1] == 'train':
        Q = np.load(data_path('q_tables', 'Qvals.npy')) 
        
        # Q_old = np.load(data_path('q_tables', 'Q_last_lr_0.6.npy'))  # 3.3.2-1
        Q = prefill_Q_heuristic(Q, r=25, rv=2) # 3.3.2-2
        # Q = prefill_Q_simple(Q, r=25, rv=2) # 3.3.2-3
        # Q = prefill_Q_gap_center_bins(Q, r=25, rv=2)
        # Q = 0.7 * Q + 0.3 * Q_old # 3.3.2-4

        print("Q shape:", Q.shape)
        print("Q min/max:", Q.min(), Q.max())
        print("Q mean/std:", Q.mean(), Q.std())
        print("Nonzero:", np.count_nonzero(Q))
        # flap vs noflap 的整体偏好：正数表示更偏 flap
        pref = Q[...,1] - Q[...,0]
        print("pref (Q1-Q0) min/max/mean:", pref.min(), pref.max(), pref.mean())
        # 看看有多少状态更偏 flap
        print("frac prefer flap:", np.mean(pref > 0))


        FPS = 1500
        screen = False
        print('Starting to train!')
    else:
        print('Invalid command line arguments. The first one should be either \'train\' or \'play\'!')
        print('If you want to load a specific Q matrix, the second one should be a filename.')
        exit()
elif(len(argv)) == 3:
    if argv[1] == 'play' and isinstance(argv[2], str):
        FPS = 30
        screen = True
        try:
            Q = np.load(input_path(argv[2]))
        except:
            print('Failed loading file!')
            exit()
    elif argv[1] == 'train' and isinstance(argv[2], str):
        FPS = 1500
        screen = False
        try:
            Q = np.load(input_path(argv[2]))
        except:
            print('Failed loading file!')
            exit()
        print('Starting to train!')
    else:
        print('Invalid command line arguments. The first one should be either \'train\' or \'play\'!')
        print('If you want to load a specific Q matrix, the second one should be a filename.')
        exit()

# list of all possible players (tuple of 3 positions of flap)
def configure_experiment(config, mode, run_dir=None):
    global Q, FPS, screen, MAX_EPISODES, SAVE_EVERY, RUN_DIR, LEARNING_ENABLED
    global SAMPLE_T, GAMMA, EPSILON, EPSILON_START, EPSILON_END, EPSILON_DECAY, LEARNING_RATE_INITIAL, LEARNING_RATE_MIN
    global LEARNING_RATE_DECAY, DISCRETIZATION_R, DISCRETIZATION_RV
    global PIPE_PASSED_REWARD, DID_NOT_DIE_REWARD, DIE_REWARD, PLAYER_FLAP_ACC, ACTIVE_CONFIG

    reset_run_state()
    ACTIVE_CONFIG = dict(config)
    RUN_DIR = Path(run_dir) if run_dir is not None else None
    episodes_key = 'training_episodes' if mode == 'train' else 'evaluation_episodes'
    MAX_EPISODES = int(config.get(episodes_key, config.get('max_episodes', MAX_EPISODES)))
    SAVE_EVERY = int(config.get('save_every', SAVE_EVERY))
    SAMPLE_T = int(config.get('sample_t', SAMPLE_T))
    GAMMA = float(config.get('gamma', GAMMA))
    EPSILON_START = float(config.get('epsilon_start', config.get('epsilon', EPSILON_START)))
    EPSILON_END = float(config.get('epsilon_end', EPSILON_END))
    EPSILON_DECAY = float(config.get('epsilon_decay', EPSILON_DECAY))
    EPSILON = EPSILON_START
    LEARNING_RATE_INITIAL = float(config.get('learning_rate', config.get('learning_rate_initial', LEARNING_RATE_INITIAL)))
    LEARNING_RATE_MIN = float(config.get('learning_rate_min', LEARNING_RATE_MIN))
    LEARNING_RATE_DECAY = float(config.get('learning_rate_decay', LEARNING_RATE_DECAY))
    DISCRETIZATION_R = int(config.get('r', DISCRETIZATION_R))
    DISCRETIZATION_RV = int(config.get('rv', DISCRETIZATION_RV))
    PIPE_PASSED_REWARD = float(config.get('pipe_passed_reward', PIPE_PASSED_REWARD))
    DID_NOT_DIE_REWARD = float(config.get('did_not_die_reward', DID_NOT_DIE_REWARD))
    DIE_REWARD = float(config.get('die_reward', DIE_REWARD))
    PLAYER_FLAP_ACC = float(config.get('player_flap_acc', PLAYER_FLAP_ACC))
    FPS = int(config.get('fps_train' if mode == 'train' else 'fps_eval', FPS))
    screen = bool(config.get('render', False))
    LEARNING_ENABLED = mode == 'train'

    Q = np.load(input_path(config.get('input_q_table', 'data/q_tables/Qvals.npy')))
    init = config.get('q_init', 'none')
    if mode == 'train':
        if init == 'heuristic':
            Q = prefill_Q_heuristic(Q, r=DISCRETIZATION_R, rv=DISCRETIZATION_RV)
        elif init == 'simple':
            Q = prefill_Q_simple(Q, r=DISCRETIZATION_R, rv=DISCRETIZATION_RV)
        elif init == 'gap_center_bins':
            Q = prefill_Q_gap_center_bins(Q, r=DISCRETIZATION_R, rv=DISCRETIZATION_RV)


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def summarize_metrics(mode, started_at, ended_at):
    rewards = np.array(episode_rewards, dtype=float)
    scores = np.array(EPISODE_SCORES, dtype=float)
    last_100 = rewards[-100:] if len(rewards) else rewards
    return {
        'status': 'ok',
        'mode': mode,
        'episodes': int(len(rewards)),
        'training_episodes': int(len(rewards)) if mode == 'train' else 0,
        'evaluation_episodes': int(len(rewards)) if mode == 'eval' else 0,
        'seed': ACTIVE_CONFIG.get('seed') if ACTIVE_CONFIG is not None else None,
        'training_time': float(ended_at - started_at) if mode == 'train' else 0.0,
        'duration_seconds': float(ended_at - started_at),
        'mean_reward': float(rewards.mean()) if len(rewards) else 0.0,
        'last_100_mean_reward': float(last_100.mean()) if len(last_100) else 0.0,
        'best_reward': float(rewards.max()) if len(rewards) else 0.0,
        'mean_score': float(scores.mean()) if len(scores) else 0.0,
        'std_score': float(scores.std()) if len(scores) else 0.0,
        'best_score': int(scores.max()) if len(scores) else 0,
        'max_score': int(scores.max()) if len(scores) else 0,
        'nonzero_q_values': int(np.count_nonzero(Q)),
        'q_min': float(Q.min()),
        'q_max': float(Q.max()),
        'q_mean': float(Q.mean()),
        'hyperparameters': {
            'learning_rate': LEARNING_RATE_INITIAL,
            'learning_rate_min': LEARNING_RATE_MIN,
            'learning_rate_decay': LEARNING_RATE_DECAY,
            'discount_factor': GAMMA,
            'epsilon_start': EPSILON_START,
            'epsilon_end': EPSILON_END,
            'epsilon_decay': EPSILON_DECAY,
            'sample_t': SAMPLE_T,
            'r': DISCRETIZATION_R,
            'rv': DISCRETIZATION_RV,
        },
        'progress': PROGRESS_METRICS,
    }


def save_run_outputs(mode, started_at, ended_at):
    metrics = summarize_metrics(mode, started_at, ended_at)
    if RUN_DIR is not None:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        np.save(RUN_DIR / 'q_table.npy', Q)
        np.save(RUN_DIR / 'rewards.npy', np.array(episode_rewards, dtype=float))
        np.save(RUN_DIR / 'average_rewards.npy', np.array(ave_rewards, dtype=float))
        write_json(RUN_DIR / 'metrics.json', metrics)
    return metrics


def run_configured_experiment(config, mode, run_dir=None):
    if 'seed' in config and config['seed'] is not None:
        random.seed(int(config['seed']))
        np.random.seed(int(config['seed']))
    configure_experiment(config, mode, run_dir)
    started_at = time.time()
    try:
        main()
    finally:
        pygame.quit()
    ended_at = time.time()
    return save_run_outputs(mode, started_at, ended_at)


# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        asset_path('assets/sprites/redbird-upflap.png'),
        asset_path('assets/sprites/redbird-midflap.png'),
        asset_path('assets/sprites/redbird-downflap.png'),
    ),
    # blue bird
    (
        asset_path('assets/sprites/bluebird-upflap.png'),
        asset_path('assets/sprites/bluebird-midflap.png'),
        asset_path('assets/sprites/bluebird-downflap.png'),
    ),
    # yellow bird
    (
        asset_path('assets/sprites/yellowbird-upflap.png'),
        asset_path('assets/sprites/yellowbird-midflap.png'),
        asset_path('assets/sprites/yellowbird-downflap.png'),
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    asset_path('assets/sprites/background-day.png'),
    asset_path('assets/sprites/background-night.png'),
)

# list of pipes
PIPES_LIST = (
    asset_path('assets/sprites/pipe-green.png'),
    asset_path('assets/sprites/pipe-red.png'),
)


try:
    xrange
except NameError:
    xrange = range





def main():

    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load(asset_path('assets/sprites/0.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/1.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/2.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/3.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/4.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/5.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/6.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/7.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/8.png')).convert_alpha(),
        pygame.image.load(asset_path('assets/sprites/9.png')).convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load(asset_path('assets/sprites/gameover.png')).convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load(asset_path('assets/sprites/message.png')).convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load(asset_path('assets/sprites/base.png')).convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    try:
        SOUNDS['die']    = pygame.mixer.Sound(asset_path('assets/audio/die' + soundExt))
        SOUNDS['hit']    = pygame.mixer.Sound(asset_path('assets/audio/hit' + soundExt))
        SOUNDS['point']  = pygame.mixer.Sound(asset_path('assets/audio/point' + soundExt))
        SOUNDS['swoosh'] = pygame.mixer.Sound(asset_path('assets/audio/swoosh' + soundExt))
        SOUNDS['wing']   = pygame.mixer.Sound(asset_path('assets/audio/wing' + soundExt))
    except pygame.error:
        SOUNDS['die'] = SOUNDS['hit'] = SOUNDS['point'] = SOUNDS['swoosh'] = SOUNDS['wing'] = SilentSound()

    while counter < MAX_EPISODES:
    # while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            #if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        if True:
            # make first flap sound and return values for mainGame
            # SOUNDS['wing'].play()
            return {
                'playery': playery + playerShmVals['val'],
                'basex': basex,
                'playerIndexGen': playerIndexGen,
            }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(movementInfo):
    global olddx, olddy, oldvy, oldflap, reward
    global counter, rewsum, highscore, totscore
    global screen

    score = playerIndex = loopIter = 0
    episode_reward = 0 #Plot
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 0, 'y': newPipe1[0]['y']}, # changed this from +200 to +0
        {'x': SCREENWIDTH + 0 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 0, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 0 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    #playerFlapAcc =  -14   # players speed on flapping
    playerFlapAcc = PLAYER_FLAP_ACC  # players speed on flapping
    playerFlapped = False # True when player flaps


    while True:

        rewsum += reward
        episode_reward += reward #Plot

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY += playerFlapAcc
                    playerFlapped = True

        traj = [(0,0), (0,0)]

        y = playery
        vy = playerVelY


        ##############################################################################
        ##############################################################################
        ##############################################################################
        #                      interesting code begins here                          #
        ##############################################################################
        ##############################################################################
        ##############################################################################



        # Measuring distance to the next pipe

        # dx is the x-axis distance from the player to the center of the nearest pipe
        # we first assume that pipe 0 is the closest
        dx = lowerPipes[0]['x'] - PLAYERX
        if dx < 0:
            # if we've passed pipe 0, pipe 1 actually the one we want to look at
            dx = lowerPipes[1]['x'] - PLAYERX

            # and we want to look at the y-axis distance from the pipe as well
            dy = lowerPipes[1]['y'] - (PIPEGAPSIZE//2) - (BIRDDIAMETER//2) - y 
        else:
            # this branch gives us the y-axis distance for pipe 0
            dy = lowerPipes[0]['y'] - (PIPEGAPSIZE//2) - (BIRDDIAMETER//2) - y

        dy = int(dy)
        dx = int(dx)

        # rewards
        pipePassedReward = PIPE_PASSED_REWARD
        didNotDieReward = DID_NOT_DIE_REWARD
        dieReward = DIE_REWARD


        # loopIter is a frame counter going from 0 to 30, and then resetting
        # every now and then, we want to take an action
        # this might be every frame, might be every 15 frames...
        sampleT = SAMPLE_T

        if loopIter % sampleT == 0:

            # ni is the learning rate
            # ni = 0.4
            ni = max(LEARNING_RATE_MIN, LEARNING_RATE_INITIAL * (LEARNING_RATE_DECAY ** counter))
            # ni = max(0.1, 0.7 * (0.999 ** counter))
            # ni = max(0.1, 0.4 * (0.999 ** counter))
            # ni = max(0.1, 0.9 * (0.999 ** counter))
            # r is the discretisation rate for (dx, dy)
            r = DISCRETIZATION_R
            # rv is the discretisation rate for vy
            rv = DISCRETIZATION_RV

            # s_t represents the last state we were in
            # s_tp is the state we're in now (the one we want to max over)
            s_t =  (int(olddx//r), int((olddy + 512)//r), int((oldvy + 50)//rv), int(oldflap))
            s_tp = (int(dx//r)   , int((dy + 512)//r)   , int((vy + 50)//rv))

            # Q update equation
            if LEARNING_ENABLED:
                Q[s_t] = (1 - ni)*Q[s_t] + ni*(reward + GAMMA*np.max(Q[s_tp]))

            # epsilon-greedy step
            if LEARNING_ENABLED:
                eps = EPSILON_END + (EPSILON_START - EPSILON_END) * (EPSILON_DECAY ** counter)
                eps = min(max(eps, min(EPSILON_START, EPSILON_END)), max(EPSILON_START, EPSILON_END))
            else:
                eps = 0.0
            if np.random.random() <= eps:
                flap = np.random.choice([True, False])
            else:
                flap = bool(np.argmax(Q[s_tp]))


            # reset the reward
            reward = 0

            # save the state info for the next loop
            olddx = dx
            olddy = dy
            oldvy = vy
            oldflap = int(flap)
        else:
            flap = False



        ##############################################################################
        ##############################################################################
        ##############################################################################
        #                      interesting code ends here                            #
        ##############################################################################
        ##############################################################################
        ##############################################################################


        if flap:
            playerVelY += playerFlapAcc
            playerFlapped = True

        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)

         
        if crashTest[0]:
            reward += dieReward
            episode_reward += dieReward #Plot

            episode_rewards.append(episode_reward) #Plot
            EPISODE_SCORES.append(score)

            counter += 1
            if SAVE_EVERY > 0 and counter % SAVE_EVERY == 0:
                print("_________________________________________________")
                print("Round", counter)
                print("Average reward in last", SAVE_EVERY, "runs:", rewsum/SAVE_EVERY)
                print("Nonzero Q values", np.count_nonzero(Q))
                print("Avg score, high score:", totscore/SAVE_EVERY, highscore)

                avg100 = rewsum / SAVE_EVERY
                ave_rewards.append(avg100)
                PROGRESS_METRICS.append({
                    'episode': counter,
                    'average_reward': float(avg100),
                    'average_score': float(totscore / SAVE_EVERY),
                    'high_score': int(highscore),
                    'nonzero_q_values': int(np.count_nonzero(Q)),
                })
                if RUN_DIR is not None:
                    RUN_DIR.mkdir(parents=True, exist_ok=True)
                    np.save(output_path('average_rewards.npy'), np.array(ave_rewards, dtype=float))
                else:
                    (DATA_DIR / 'prefill').mkdir(parents=True, exist_ok=True)
                    (DATA_DIR / 'q_tables').mkdir(parents=True, exist_ok=True)
                    (DATA_DIR / 'rewards').mkdir(parents=True, exist_ok=True)
                    np.save(data_path('prefill', 'prefill_heuristic.npy'), np.array(ave_rewards, dtype=float))

                rewsum = 0
                highscore = 0
                totscore = 0
                if RUN_DIR is not None:
                    np.save(output_path('q_table.npy'), Q)
                    np.save(output_path('rewards.npy'), np.array(episode_rewards, dtype=float)) #Plot #save_rewards
                else:
                    np.save(data_path('q_tables', 'Q_last_pre_heuristic.npy'), Q)
                    np.save(data_path('rewards', 'rewards_pre_heuristic.npy'), np.array(episode_rewards, dtype=float)) #Plot #save_rewards
                
                if RUN_DIR is not None:
                    print('Saved progress in', RUN_DIR)
                else:
                    print('Saved progress in data/q_tables and data/rewards')

            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot
            }

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                reward += pipePassedReward

                totscore += 1
                if score > highscore:
                    highscore = score
                # SOUNDS['point'].play()
            else:
                reward += didNotDieReward


        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        #if playerVelY < playerMaxVelY and not playerFlapped:
        playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        # hello
        if screen:
            SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))


        # hello
        if screen:
            SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot
        
        playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (playerx, playery))



        playerOffsetX = IMAGES['player'][0].get_width() / 2
        playerOffsetY = IMAGES['player'][0].get_height() / 2

        # hello
        if screen:
            pygame.draw.lines(SCREEN, (255,0,0), False, [(x+playerOffsetX,y+playerOffsetY) for (x,y) in traj], 3)
            pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    #SOUNDS['hit'].play()
    #if not crashInfo['groundCrash']:
    #    SOUNDS['die'].play()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            #if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        if True:
            if playery + playerHeight >= BASEY - 1:
                return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        #SCREEN.blit(IMAGES['background'], (0,0))

        #for uPipe, lPipe in zip(upperPipes, lowerPipes):
        #    SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
        #    SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        #SCREEN.blit(IMAGES['base'], (basex, BASEY))
        #showScore(score)

        


        #playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        #SCREEN.blit(playerSurface, (playerx,playery))
        #SCREEN.blit(IMAGES['gameover'], (50, 180))

        #FPSCLOCK.tick(FPS)
        #pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    #gapY = random.randrange(int(BASEY * 0.5 - PIPEGAPSIZE), int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground or ceiling
    if player['y'] + player['h'] >= BASEY - 1 or player['y'] + player['h'] <= 0:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
