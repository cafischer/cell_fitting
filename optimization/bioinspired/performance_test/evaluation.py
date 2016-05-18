import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from optimization.bioinspired.problem import Problem
from optimization.fitfuns import run_simulation
from errfuns import errfun_pointtopoint, errfun_featurebased
import os.path
import json

__author__ = 'caro'


method = 'DEA'

# read in observed data from optimization
path = './results/'+method+'/individuals_file.csv'
population_file = pd.read_csv(path, names=['generation', 'index', 'fitness', 'candidate'], header=None)

# estimate error landscape
with open('./results/params.json') as f:
    params = json.load(f)
p1_range = np.arange(params['lower_bound'], params['upper_bound']+0.0001, 0.01)
p2_range = np.arange(params['lower_bound'], params['upper_bound']+0.0001, 0.01)

if os.path.isfile('./results/error_landscape.npy'):
    error = np.load('./results/error_landscape.npy')
else:
    candidates = list()
    for i, p1 in enumerate(p1_range):
        for j, p2 in enumerate(p2_range):
            candidates.append([p1, p2])
    problem = Problem(params)
    error = problem.evaluator(candidates, [])
    error = np.reshape(error, (len(p1_range), len(p2_range)))
    with open('./results/error_landscape.npy', 'w') as f:
        np.save(f, error)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=50)
    X, Y = np.meshgrid(p1_range, p2_range)
    surf = ax.plot_surface(X, Y, error.T, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.6,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()


def init():
    title.set_text('Plot of candidate solutions.')
    return points, title

def update_points(generation, points, points_data):
    points.set_data(points_data[generation][:, 0], points_data[generation][:, 1])
    points.set_3d_properties(points_data[generation][:, 2])
    title.set_text('Candidates in generation: '+str(generation))
    return points, title


points_data = list()
population_size = len(population_file.loc[population_file.generation == 0])
n_generations = population_file.generation.iloc[-1] + 1
for generation in range(n_generations):
    points_data.append(np.zeros([population_size, 3]))
    generation_file = population_file.loc[population_file.generation == generation]
    for i in range(population_size):
        candidate_file = generation_file.iloc[i]
        vars = candidate_file['candidate'][2:-1]
        [p1, p2] = map(float, vars.split())
        points_data[generation][i, :] = np.array([p1, p2, candidate_file['fitness']])

# animated plot
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')
ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('')
ax.view_init(elev=20, azim=50)
title = ax.title
X, Y = np.meshgrid(p1_range, p2_range)
surf = ax.plot_surface(X, Y, error.T, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.6,
                   linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.9, aspect=10)
points = ax.plot([], [], [], 'ok')[0]
line_ani = animation.FuncAnimation(fig, update_points, n_generations, init_func=init, fargs=(points, points_data),
                                   interval=500, blit=False, repeat=True)
plt.show()
