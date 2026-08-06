import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

N = 24 
path = cvx.Variable((N, 2))
flap = cvx.Variable(N-1, boolean=True)
last_solution = [False, False, False]
last_path = [(0,0),(0,0)]

PIPEGAPSIZE  = 100
PIPEWIDTH = 52
BIRDWIDTH = 34
BIRDHEIGHT = 24
BIRDDIAMETER = np.sqrt(BIRDHEIGHT**2 + BIRDWIDTH**2)
SKY = 0
GROUND = (512*0.79)-1
PLAYERX = 57


def getPipeConstraintsDistance(x, y, lowerPipes):
    constraints = []
    pipe_dist = 0
    for pipe in lowerPipes:
        dist_from_front = pipe['x'] - x - BIRDDIAMETER
        dist_from_back = pipe['x'] - x + PIPEWIDTH
        if (dist_from_front < 0) and (dist_from_back > 0):
            constraints += [y <= (pipe['y'] - BIRDDIAMETER)]
            constraints += [y >= (pipe['y'] - PIPEGAPSIZE)]
            pipe_dist += cvx.abs(pipe['y'] - (PIPEGAPSIZE//2) - (BIRDDIAMETER//2) - y)
    return constraints, pipe_dist

def solve(playery, playerVelY, lowerPipes, playerAccY):

    # speed in the x-direction
    pipeVelX = -4 
    # acceleration gained by flapping
    playerFlapAcc = -14

    # unpack path variables
    y = path[:,0]
    vy = path[:,1]

    # c is the constraint list
    c = []
    # don't hit the limits of the screen
    c += [y <= GROUND, y >= SKY]
    # initial conditions
    c += [y[0] == playery, vy[0] == playerVelY]

    x = PLAYERX
    xs = [x] 

    # look ahead
    for t in range(N-1):

        # update x-position
        x -= pipeVelX
        xs += [x]

        # add constraints representing the model dynamics
        c += [vy[t + 1] ==  vy[t] + playerAccY  + playerFlapAcc * flap[t] ]
        c += [y[t + 1] ==  y[t] + vy[t + 1]]

        # add pipe constraints 
        # distance from the center of the pipe-gap is stored in the dist variable
        pipe_c, dist = getPipeConstraintsDistance(x, y[t+1], lowerPipes)
        c += pipe_c


    #############################################
    ### this is where you enter the objective ###
    #############################################

    objective = cvx.Minimize( ??? )





    prob = cvx.Problem(objective, c)
    try:
        prob.solve(verbose = False)
        last_path = list(zip(xs, y.value))
        last_solution = np.round(flap.value).astype(bool)
        return last_solution[0], last_path
    except:
        try:
            last_solution = last_solution[1:]
            last_path = [((x-4), y) for (x,y) in last_path[1:]]
            return last_solution[0], last_path
        except:
            return False, [(0,0), (0,0)]