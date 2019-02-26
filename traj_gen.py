
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

traj_raw = scipy.io.loadmat("traj_fl_1")['traj_fl_1']
np.savez('traj_fl1', traj1=np.transpose(traj_raw))