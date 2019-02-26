'''
This code is originally from "https://github.com/studywolf/pydmps" by Studywolf, and modified for DOOSAN Project
More specific, it imitates drawing number 2.

Author: Bukun Son
Start date: 2019.Feb.21
Last date: 2019.Feb.21
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
import dmp_discrete

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

y_des = np.load('traj_fl1.npz')['traj1']
y_des -= y_des[:, 0][:, None]

# test normal run
dmp = dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=50, ay=np.ones(2)*10.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=True)
y_track, dy_track, ddy_track = dmp.rollout()
plt.figure(1, figsize=(6,6))

plt.figure(2)
plt.plot(y_track[:,0], y_track[:, 1], 'b', lw=2)
plt.title('DMP system - draw number 2')

plt.axis('equal')
#plt.xlim([6, 10])
#plt.ylim([-2, 0.5])
plt.show()
