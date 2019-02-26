"""
This code is originally from "https://github.com/studywolf/pydmps" by Studywolf, and modified for DOOSAN Project
More specific, it is for canonical system in dynamic movement primitives(DMPs) by Stefan Schaal (2002)

Author: Bukun Son
Start date: 2019.Feb.21
Last date: 2019.Feb.21
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class CanonicalSystem():
    def __init__(self, dt, ax=1.0):

        self.ax = ax                                        # a gain term on the dynamical system
        self.dt = dt                                        # the time-step
        self.run_time = 1                                 # default run time
        self.timesteps = int(self.run_time / self.dt)       #
        self.step = self.step_discrete
        self.reset_state()

    def rollout(self, **kwargs):
        # Generate x for open loop movements.
        if 'tau' in kwargs:
            timesteps = int(self.timesteps / kwargs['tau'])
        else:
            timesteps = self.timesteps

        self.x_track = np.zeros(timesteps)

        self.reset_state()
        for t in range(timesteps):
            self.x_track[t] = self.x
            self.step(**kwargs)
        return self.x_track

    def reset_state(self):
        self.x = 1.0        # Reset the system as initial state

    def step_discrete(self, tau=1.0, error_coupling=1.0):
        """
        Decaying from 1 to 0 according to dx = -ax*x.
        tau: gain on execution time
        error_coupling: slow down if the error is > 1
        """
        self.x += (-self.ax * self.x * error_coupling) * tau * self.dt
        return self.x
