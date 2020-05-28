import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pickle
import os
#import colour
import torch

# config
def cal_rew(encoder,path):
    s,a,r = path['observations'],path['actions'],path['rewards']
    s,a,r=torch.FloatTensor(s),torch.FloatTensor(a),torch.FloatTensor(r)
    input = torch.cat([s,a,r],dim=1)
    print(input.shape)
    input = torch.unsqueeze(input,0)
    output = encoder.forward_seq(input)
    print(output.shape)
    var = torch.mean(torch.log(torch.nn.functional.softplus(output[:,:,10:])),dim=2)
    var = torch.mean(torch.nn.functional.softplus(output[:,:,10:]),dim=2)
    #var = torch.log(var)
    print(var.shape)
    #print(r,var)
    return var.view(r.shape[0],r.shape[1])


exp_id = '2019_10_10_19_43_28' #pearl, sparse reward
exp_id = '2019_10_11_08_57_57' #pearl, sparse reward, large radius
tlow, thigh = 80, 100 # task ID range
# see `n_tasks` and `n_eval_tasks` args in the training config json
# by convention, the test tasks are always the last `n_eval_tasks` IDs
# so if there are 100 tasks total, and 20 test tasks, the test tasks will be IDs 81-100
epoch = 775# good
epoch = 278# good
gr = 0.3 # goal radius, for visualization purposes


expdir = './outputpearl/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
dir = './outputpearl/sparse-point-robot/{}/'.format(exp_id)
expdir = './output/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
dir = './output/sparse-point-robot/{}/'.format(exp_id)
file = './info.pkl'
#expdir = './outputiter_rew/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
#dir = './outputiter_rew/sparse-point-robot/{}/'.format(exp_id)
#expdir = './outputfin/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
#dir = './outputfin/sparse-point-robot/{}/'.format(exp_id)
#expdir = './outputfin_sparse/sparse-point-robot/{}/eval_trajectories/'.format(exp_id)
#dir = './outputfin_sparse/sparse-point-robot/{}/'.format(exp_id)
# helpers
def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


task_id = 0
data = load_pkl(file)
print(data['actions'].shape)
print(data['env_infos']['goal'])
#print(data['observations'])
observations = data['observations'][task_id,:,:2]
print(observations.shape)

goals = [[-0.5000000000000002, 0.8660254037844385], [0.766044443118978, 0.6427876096865393], [-0.9396926207859083, 0.3420201433256689], [0.654860733945285, 0.7557495743542583], [0.9994965423831851, 0.03172793349806765], [0.7237340381050701, 0.690079011482112], [-1.0, 1.2246467991473532e-16], [-0.14231483827328523, 0.9898214418809327], [0.9819286972627067, 0.18925124436041021], [0.9594929736144974, 0.28173255684142967], [-0.654860733945285, 0.7557495743542583], [-0.8888354486549234, 0.4582265217274105], [-0.8579834132349771, 0.5136773915734063], [-0.9594929736144974, 0.28173255684142967], [-0.9954719225730846, 0.09505604330418244], [-0.9500711177409454, 0.31203344569848696], [0.32706796331742155, 0.9450008187146685], [-0.975429786885407, 0.2203105327865408], [-0.3568862215918718, 0.9341478602651068], [0.7452644496757548, 0.6667690005162916]]



plt.figure(figsize=(8,8))
axes = plt.axes()
axes.set(aspect='equal')
plt.axis([-1.55, 1.55, -0.55, 1.55])
for g in goals:
    circle = plt.Circle((g[0], g[1]), radius=gr)
    axes.add_artist(circle)
rewards = 0
final_rewards = 0
num_trajs=4
cmap = matplotlib.cm.get_cmap('plasma')
sample_locs = np.linspace(0, 0.9, num_trajs)
colors = [cmap(s) for s in sample_locs]
fig, axes = plt.subplots(3, 3, figsize=(12, 20))
t = 10



for j in range(3):
    for i in range(3):
        axes[i, j].set_xlim([-1.55, 1.55])
        axes[i, j].set_ylim([-0.55, 1.55])
        for k, g in enumerate(goals):
            alpha = 1 if k == t else 0.2
            circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
            axes[i, j].add_artist(circle)
        indices = list(np.linspace(0, 4, num_trajs, endpoint=False).astype(np.int))
        counter = 0
        for idx in indices:
            states = observations[idx*32:(idx+1)*32,:]
            print(states.shape)
            axes[i, j].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[counter])
            axes[i, j].plot(states[-1, 0], states[-1, 1], '-x', markersize=10, color=colors[counter])
            axes[i, j].set(aspect='equal')
            counter += 1
        #axes[i,j].set_title("Last episode return:%f"%reward[t])
        t += 1

fig.suptitle("Point-Robot-Sparse, E-RL^2 ",size=22)

plt.show()