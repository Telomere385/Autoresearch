import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

N = 24  # look-ahead horizon
path = cvx.Variable((N, 2))
flap = cvx.Variable(N-1, boolean=True)
last_solution = [False, False, False]
last_path = [(0,0),(0,0)]

PIPEGAPSIZE  = 100
PIPEWIDTH = 52
BIRDWIDTH = 34
BIRDHEIGHT = 24
BIRDDIAGONAL = np.sqrt(BIRDHEIGHT**2 + BIRDWIDTH**2)
SKY = 0
GROUND = (512*0.79)-1
PLAYERX = 57


def getPipeConstraintsDistance(x, y, lowerPipes):
    constraints = []
    pipe_dist = 0
    margin = 0
    
    for pipe in lowerPipes:
        dist_from_front = pipe['x'] - x - BIRDDIAGONAL
        dist_from_back = pipe['x'] - x + PIPEWIDTH
        if (dist_from_front < 0) and (dist_from_back > 0):
            # constraints += [y <= (pipe['y'] - BIRDDIAGONAL)]
            # constraints += [y >= (pipe['y'] - PIPEGAPSIZE)]
            constraints += [y <= (pipe['y'] - BIRDDIAGONAL - margin )]
            constraints += [y >= (pipe['y'] - PIPEGAPSIZE  + margin )]
            pipe_dist += cvx.abs(pipe['y'] - (PIPEGAPSIZE//2) - (BIRDDIAGONAL//2) - y)
    return constraints, pipe_dist

def solve(playery, playerVelY, lowerPipes):
    pipeVelX = -4           # speed in the x-direction
    playerAccY = 1          # gravitational acceleration
    playerFlapAcc = -14     # acceleration gained by flapping

    # unpack path variables
    y = path[:,0]
    vy = path[:,1]

    constraints = []                                        # init constraint list
    constraints += [ y <= GROUND, y >= SKY ]                # don't hit the limits of the screen
    constraints += [ y[0] == playery, vy[0] == playerVelY ] # initial conditions

    x = PLAYERX
    xs = [x]    # init x list
    cost = 0

    for t in range(N-1):    # look ahead
        # update x-position
        x -= pipeVelX
        xs += [x]
        
        # distance from the center of the pipe-gap is stored in the dist variable
        # pipe_c, dist = getPipeConstraintsDistance(x, y[t+1], lowerPipes)
        # eps = cvx.Variable()  # scalar slack for this step
        # constraints += [eps <= 3, eps >= -3] # limit the slack to a reasonable range
        # pipe_c, dist = getPipeConstraintsDistance(x, y[t+1], lowerPipes, slack=eps)
        pipe_c, dist = getPipeConstraintsDistance(x, y[t+1], lowerPipes)
        # add pipe constraints 
        constraints += pipe_c 
        
        # TODO
        # add constraints representing the model dynamics
        # constraints += ...
        constraints += [ y[t+1]  == y[t] + vy[t] ]
        constraints += [ vy[t+1] == vy[t] + playerAccY + playerFlapAcc * flap[t] ]

        # increase cost
        # cost += dist + 0.08 * cvx.abs(vy[t+1])
        cost += dist 

    objective = cvx.Minimize(cost)              # define objective
    prob = cvx.Problem(objective, constraints)  # init the optimization problem

    try:
        prob.solve(verbose = False) # use this line for open source solvers
        #prob.solve(verbose = False, solver="GUROBI") # use this line if you have access to Gurobi, a faster solver
        last_path = list(zip(xs, y.value)) # store the path
        last_solution = np.round(flap.value).astype(bool) # store the solution
        
        return last_solution[0], last_path # return the next input and path for plotting

    except:
        try:
            last_solution = last_solution[1:] # if we didn't get a solution this round, use the last solution
            last_path = [((x-4), y) for (x,y) in last_path[1:]]

            return last_solution[0], last_path

        except:
            return False, [(0,0), (0,0)] # if we fail to solve many times in a row, do nothing