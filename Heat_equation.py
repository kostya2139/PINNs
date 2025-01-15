import tensorflow as tf
import numpy as np

def initial():
    np.random.seed(0)
    tf.random.set_seed(0)


class PINNs:
    def __init__(self, n_neurons, n_layers, alpha, ini, boun):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.alpha = alpha
        self.x_ini = ini[0:1, :]




def main(ini, boun):
    n_neurons = 10
    n_layers = 10
    initial_conditions, boundary_conditions = ini, boun # [x, t, u]


