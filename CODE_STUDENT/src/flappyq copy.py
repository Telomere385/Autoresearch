from itertools import cycle
from pathlib import Path
import random
import sys
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



if(len(argv)) == 2:
    if argv[1] == 'play':
        Q = np.load(data_path('q_tables', 'Q_last_lr_0.6.npy'))
        FPS = 30
        screen = True
    elif argv[1] == 'train':
        Q = np.load(data_path('q_tables', 'Qvals.npy')) 
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

    SOUNDS['die']    = pygame.mixer.Sound(asset_path('assets/audio/die' + soundExt))
    SOUNDS['hit']    = pygame.mixer.Sound(asset_path('assets/audio/hit' + soundExt))
    SOUNDS['point']  = pygame.mixer.Sound(asset_path('assets/audio/point' + soundExt))
    SOUNDS['swoosh'] = pygame.mixer.Sound(asset_path('assets/audio/swoosh' + soundExt))
    SOUNDS['wing']   = pygame.mixer.Sound(asset_path('assets/audio/wing' + soundExt))

    MAX_EPISODES = 5000 #3.3.2
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
    playerFlapAcc =  -10  # players speed on flapping
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
        pipePassedReward = 10
        didNotDieReward = 0.05
        dieReward = -50


        # loopIter is a frame counter going from 0 to 30, and then resetting
        # every now and then, we want to take an action
        # this might be every frame, might be every 15 frames...
        sampleT = 3

        if loopIter % sampleT == 0:

            # ni is the learning rate
            # ni = 0.4
            ni = max(0.1, 0.6 * (0.999 ** counter))
            # ni = max(0.1, 0.7 * (0.999 ** counter))
            # ni = max(0.1, 0.4 * (0.999 ** counter))
            # ni = max(0.1, 0.9 * (0.999 ** counter))
            # r is the discretisation rate for (dx, dy)
            r = 25
            # rv is the discretisation rate for vy
            rv = 2

            # s_t represents the last state we were in
            # s_tp is the state we're in now (the one we want to max over)
            s_t =  (olddx//r, (olddy + 512)//r, (oldvy + 50)//rv, int(oldflap))
            s_tp = (dx//r   , (dy + 512)//r   , (vy + 50)//rv)

            # Q update equation
            Q[s_t] = (1 - ni)*Q[s_t] + ni*(reward + 0.95*np.max(Q[s_tp]))

            # epsilon-greedy step
            eps = 0.001
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

            counter += 1
            if counter % 100 == 0:
                print("_________________________________________________")
                print("Round", counter)
                print("Average reward in last 100 runs:", rewsum/100)
                print("Nonzero Q values", np.count_nonzero(Q))
                print("Avg score, high score:", totscore/100, highscore)

                avg100 = rewsum / 100
                ave_rewards.append(avg100)
                (DATA_DIR / 'rewards').mkdir(parents=True, exist_ok=True)
                (DATA_DIR / 'q_tables').mkdir(parents=True, exist_ok=True)
                np.save(data_path('rewards', 'ave_reward.npy'), np.array(ave_rewards, dtype=float))

                rewsum = 0
                highscore = 0
                totscore = 0
                
                np.save(data_path('q_tables', 'Q_last.npy'), Q)
                np.save(data_path('rewards', 'rewards.npy'), np.array(episode_rewards, dtype=float)) #Plot #save_rewards
                
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
