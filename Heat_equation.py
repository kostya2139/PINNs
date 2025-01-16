import tensorflow as tf
import numpy as np

def initial():
    np.random.seed(0)
    tf.random.set_seed(0)



class PINNs:
    def __init__(self, n_neurons, n_layers, alpha, u, f):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.alpha = alpha
        self.x_u = u[:, 0:1]
        self.t_u = u[:, 1:2]
        self.u = u[:, 2:3]
        self.x_f = f[:, 0:1]
        self.t_f = f[:, 1:2]
        self.layers = [2] + n_layers * [n_neurons] + [1]

        self.weights, self.biases = self.initialize_NN(self.layers)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])        
                
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf) 

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))
               
                
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

        




    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)




def main(ini, boun):
    initial()
    n_neurons = 10
    n_layers = 10
    initial_conditions, boundary_conditions = ini, boun # [x, t, u]

def def_ini_boun(n_ini, n_boun):
    x_ini = np.random.uniform(size = (n_ini, 1))
    t_ini = np.zeros((n_ini,1))
    x_boun = np.random.choice([0, 1], (n_boun,1))
    t_boun = np.random.uniform(size = (n_boun,1))

    return (np.hstack((x_ini, t_ini)), np.hstack((x_boun, t_boun)))


print(def_ini_boun(10, 10))
