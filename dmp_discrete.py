"""
This code is originally from "https://github.com/studywolf/pydmps" by Studywolf, and modified for DOOSAN Project
More specific, it is for DMPs in discrete time(2002)

Author: Bukun Son
Start date: 2019.Feb.21
Last date: 2019.Feb.21
"""

from dmp import DMP
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


class DMPs_discrete(DMP):
    def __init__(self, **kwargs):
        # call super class constructor
        # dmp.py 의 부모클래스를 상속
        super(DMPs_discrete, self).__init__(**kwargs)

        # set center of Gaussian basis functions
        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax
        self.check_offset()

    def gen_centers(self):
        # Set the center of the Gaussian basis functions be spaced evenly
        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.ones(len(des_c))

        for n in range(len(des_c)):
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        # Generates the diminishing front term on the forcing term.
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        # Generates the goal for path
        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        # Generates the activity of the basis functions
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        # Generate a set of weights over the basis functions

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)

# ==============================
# Test code
# ==============================
if __name__ == "__main__":

    # test normal run
    dmp = DMPs_discrete(dt=.01, n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    y_track, dy_track, ddy_track = dmp.rollout()

    """
    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track))*dmp.goal, 'r--', lw=2)
    plt.plot(y_track, lw=2)
    plt.title('DMP system - no forcing term')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['goal', 'system state'], loc='lower right')
    plt.tight_layout()
    """

    # test imitation of path run
    plt.figure(2, figsize=(6, 4))
    n_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    path1 = np.sin(np.arange(0, 1, .01))

    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.):] = .5

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path (y_des=np.array([path1, path2]))
        # change the scale of the movement
        dmp.goal[0] = 0.1
        dmp.goal[1] = 5

        y_track, dy_track, ddy_track = dmp.rollout(tau=3)

        plt.figure(2)
        plt.subplot(211)
        plt.plot(y_track[:, 0], lw=2)
        plt.subplot(212)
        plt.plot(y_track[:, 1], lw=2)


    # Plot

    plt.subplot(211)
    a = plt.plot(path1 / path1[-1] * dmp.goal[0], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend([a[0]], ['desired path'], loc='lower right')
    plt.subplot(212)
    b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['%i BFs' % i for i in n_bfs], loc='lower right')

    plt.tight_layout()
    plt.show()
