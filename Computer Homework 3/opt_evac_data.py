# data for optimal evacuation problem
import numpy as np

T = 30
A = np.array([[-1., -1., 0., 0., 0., 0., 0., 0., 0.],
              [1., 0., -1., 0., 0., 0., 0., 0., 0.],
              [0., 0., 1., -1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., -1., -1., 0., 0., 0.],
              [0., 0., 0., 0., 1., 0., -1., 0., 0.],
              [0., 0., 0., 1., 0., 0., 1., -1., 0.],
              [0., 0., 0., 0., 0., 1., 0., 0., -1.],
              [0., 0., 0., 0., 0., 0., 0., 1., 1.]])
Q = np.array([1., 1., 1., 1., 1., 0.8, 1., 0.4])
F = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
q1 = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
r = np.array([1., 0.2, 0.2, 0.5, 0.5, 0., 0.5, 0.])
s = np.array([1., 0.2, 0.2, 0.5, 0.5, 0., 0.5, 0.])
rtild = np.array([0.1, 0.2, 0.1, 5., 0.4, 0.2, 0.4, 0.4, 0.2])
stild = np.array([2.8, 5.6, 2.8, 140., 11.2, 5.6, 11.2, 11.2, 5.6])
