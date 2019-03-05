"""
This code is originally from "https://github.com/studywolf/pydmps" by Studywolf, and modified for DOOSAN Project
More specific, it is for task parameterized dynamic movement primitives(TP-DMPs) by Stefan Schaal (2002)

Author: Bukun Son
Start date: 2019.Feb.28
Last date: 2019.March.1

"""

import numpy as np
from cs import CanonicalSystem
import scipy.interpolate
import warnings
import matplotlib.pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from keras import layers
from keras import Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

class DMPs:
    def __init__(self, n_dmps, dt=.01, y0=0, goal=1, ay=None, by=None, **kwargs):
        self.n_dmps = n_dmps                # number of dynamic motor primitives
        self.dt = dt                        # time-step for simulation

        # set the initial state
        y0 = np.ones(self.n_dmps)*y0
        self.y0 = y0

        # set the goal state
        goal = np.ones(self.n_dmps)*goal
        self.goal = goal

        # set the constant gain on attractor
        self.ay = np.ones(n_dmps) * 25. if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4. if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)     # generate canonical system
        self.timesteps = int(self.cs.run_time / self.dt)    # number of time steps

        # set up the DMP system
        self.reset_state()

        self.model = self.build_model()

    def force_target(self, y_des):
        # set initial state and goal

        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)
        #self.check_offset()

        # generate function to interpolate the desired trajectory
        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])
        #f3 = plt.figure(3)
        #plt.plot(x)
        #f3.show(x)
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)

        y_des = path

        # calculate velocity of y_des
        dy_des = np.diff(y_des) / self.dt
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self.n_dmps, 1)), dy_des))

        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self.n_dmps, 1)), ddy_des))

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = (ddy_des[d] - self.ay[d] *
                              (self.by[d] * (self.goal[d] - y_des[d]) -
                              dy_des[d]))
        return f_target

    def gen_goal(self, y_des):
        return np.copy(y_des[:, -1])

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.
        tau float: scales the time step (increase tau to make the system execute faster)
        error float: optional system feedback
        """
        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        for d in range(self.n_dmps):
            # generate the forcing term
            #f = self.model.predict()
            f=0
            # DMP acceleration
            #self.ddy[d] = (self.ay[d] *(self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]/tau) + f) * tau
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * self.dt * error_coupling

        return self.y, self.dy, self.ddy


    def rollout(self, timesteps=None, **kwargs):
        """Generate a system trial, no feedback is incorporated."""

        self.reset_state()

        if timesteps is None:
            if 'tau' in kwargs:
                timesteps = int(self.timesteps / kwargs['tau'])
            else:
                timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def build_model(self):
        model_in = Input(shape=(1,), name='joint')      # input shape is canonical system param + num of task params.
        dense1 = layers.Dense(30)(model_in)
        batch1 = layers.BatchNormalization()(dense1)
        act1 = layers.Activation(activation='relu')(batch1)
        dense2 = layers.Dense(30)(act1)
        batch2 = layers.Activation(activation='relu')(dense2)
        act2 = layers.Activation(activation='relu')(batch2)
        answer = layers.Dense(self.n_dmps)(act2)

        final_model = Model(model_in, answer)

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        final_model.compile(optimizer=adam, loss='mse', metrics=['mse'])
        print(final_model.summary())

        return final_model

    def imit_path(self, y_des, plot=True):
        early_stop = EarlyStopping(patience=10)
        input1 = self.cs.rollout()

        history = self.model.fit(input1, self.force_target(y_des),
                                 batch_size=100, epochs=200000, validation_split=0, shuffle=True, callbacks=None)


        mse_history = history.history['mean_squared_error']
        loss_history = history.history['loss']

        if plot is True:
            plt.figure(2)
            #plt.subplot(1, 2, 1)
            plt.plot(range(1, len(mse_history) + 1), mse_history)
            plt.title('Model mean abolute error')
            plt.xlabel('Epochs')
            plt.ylabel('Validation MAE')

        self.reset_state()


    '''
      def check_offset(self):
          """Check to see if initial position and goal are the same
          if they are, offset slightly so that the forcing term is not 0"""

          for d in range(self.n_dmps):
              if (self.y0[d] == self.goal[d]):
                  self.goal[d] += 1e-4
      '''


# ==============================
# Test code
# ==============================
if __name__ == "__main__":

    dmp = DMPs(n_dmps=1, dt=.01)
    y_track, dy_track, ddy_track = dmp.rollout()


    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(y_track)) * dmp.goal, 'r--', lw=2)
    plt.plot(y_track, lw=2)
    plt.title('DMP system - no forcing term')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['goal', 'system state'], loc='lower right')
    plt.tight_layout()


    # a straight line to target
    path1 = np.sin(np.arange(0, 1, .01))

    # change the scale of the movement
    #dmp.goal[0] = 1
    dmp.imit_path(y_des=np.array(path1))

    y_track, dy_track, ddy_track = dmp.rollout(tau=1)

    plt.figure(3)
    plt.plot(y_track[:,0], lw=2)

    plt.show()
