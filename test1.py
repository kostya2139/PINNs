import numpy as np
import tensorflow as tf
from pyDOE import lhs

def initial():
    np.random.seed(1234)
    tf.random.set_seed(1234)

# Define the PINN model
class HeatEquationPINN():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2,)),  # Input layer for (x, t)
            tf.keras.layers.Dense(50, activation='tanh'),  # First hidden layer
            tf.keras.layers.Dense(50, activation='tanh'),  # Second hidden layer
            tf.keras.layers.Dense(1)  # Output layer: predicts u(x, t)
        ])
        
    
    def __call__(self, x, t):
        return self.model(x, t)

# PDE residual
def pde_residual(model, x, t, alpha):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(x, t)
        u_t = tape.gradient(u, t)
    u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    del tape
    return u_t - alpha * u_xx

# Loss function
def loss_function(model, x, t, u_true, alpha):
    pde_loss = tf.reduce_mean(tf.square(pde_residual(model, x, t, alpha)))
    boundary_loss = ...  # Add boundary condition loss
    initial_loss = ...   # Add initial condition loss
    return pde_loss + boundary_loss + initial_loss

# Training loop
def train(model, x, t, u_true, alpha, epochs, lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_function(model, x, t, u_true, alpha)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

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
    u_boun = np.zeros((n_ini, 1))
    u_ini = np.sin(2* np.pi * x_ini)


    return (np.hstack((x_ini, t_ini, u_ini)), np.hstack((x_boun, t_boun, u_boun)))




# Example usage
alpha = 0.01
x = np.random.uniform(0, 1, (1000, 1))
t = np.random.uniform(0, 1, (1000, 1))
u_true = np.sin(np.pi * x) * np.exp(-np.pi**2 * t)  # Example analytical solution
model = HeatEquationPINN()
train(model, x, t, u_true, alpha, epochs=10000, lr=1e-3)