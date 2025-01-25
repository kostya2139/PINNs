import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools

#Constant
alpha = 0.2

# Network
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),  # Input: (x, t)
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)  # Output: u(x, t)
    ])
    return model


model = create_model()

# Derivatives
def pde_loss(x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
    
    u_xx = tape.gradient(u_x, x)
    del tape

    # Heat equation
    residual = u_t - alpha * u_xx
    return tf.reduce_mean(tf.square(residual))

# Loss initial and boundary
def ic_loss(x_ic, t_ic, u_ic):
    u_pred = model(tf.concat([x_ic, t_ic], axis=1))
    return tf.reduce_mean(tf.square(u_pred - u_ic))

def bc_loss(x_bc, t_bc, u_bc):
    u_pred = model(tf.concat([x_bc, t_bc], axis=1))
    return tf.reduce_mean(tf.square(u_pred - u_bc))

# Training data
N_collocation = 10000  
N_ic = 100  
N_bc = 100  

# Collocation points 
x_collocation = np.random.uniform(0, 1, (N_collocation, 1)).astype(np.float32)
t_collocation = np.random.uniform(0, 1, (N_collocation, 1)).astype(np.float32)

# Initial condition
x_ic = np.random.uniform(0, 1, (N_ic, 1)).astype(np.float32)
t_ic = np.zeros((N_ic, 1), dtype=np.float32)  # At t=0
u_ic = np.exp(-((x_ic - 0.5)**2) / 0.01)  # Gaussian initial condition

# Boundary conditions: u(0, t) = 0, u(1, t) = 0
t_bc = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
x_bc_0 = np.zeros((N_bc, 1), dtype=np.float32)
x_bc_1 = np.ones((N_bc, 1), dtype=np.float32)
u_bc = np.zeros((N_bc, 1), dtype=np.float32)

# Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def train_step(x_collocation, t_collocation, x_ic, t_ic, u_ic, x_bc_0, x_bc_1, t_bc):
    with tf.GradientTape() as tape:
        u_pred_collocation = model(tf.concat([x_collocation, t_collocation], axis=1))
        loss_pde = pde_loss(x_collocation, t_collocation)

        loss_ic = ic_loss(x_ic, t_ic, u_ic)
        loss_bc_0 = bc_loss(x_bc_0, t_bc, u_bc)
        loss_bc_1 = bc_loss(x_bc_1, t_bc, u_bc)

        total_loss = loss_pde + loss_ic + loss_bc_0 + loss_bc_1

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

x_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
t_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)

X, T = np.meshgrid(x_test, t_test)
X_flat = X.flatten().reshape(-1, 1).astype(np.float32)
T_flat = T.flatten().reshape(-1, 1).astype(np.float32)


# Training loop (same as before)
epochs = 1000
for epoch in range(epochs):
    loss = train_step(x_collocation, t_collocation, x_ic, t_ic, u_ic, x_bc_0, x_bc_1, t_bc)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4e} and second loss {np.linalg.norm(model.predict(np.hstack((X_flat, T_flat))) - np.exp(-((X_flat - 0.5)**2) / 0.1))}")

# Visualization

u_pred = model.predict(np.hstack((X_flat, T_flat)))
U_pred = u_pred.reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.contourf(X, T, U_pred, levels=50, cmap="viridis")
plt.colorbar(label="u(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Solution of the Heat Equation via PINN")
plt.show()



#u_pred = model.predict(np.hstack((X_flat, T_flat)))
#U_pred = u_pred.reshape(100, 100)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, T, U_pred, cmap="viridis", edgecolor='none')


ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x, t)")
ax.set_title("Solution of the Heat Equation via PINN")


plt.show()


print(np.linalg.norm(u_pred - np.exp(-((X_flat - 0.5)**2) / 0.1)))  # Error between predicted and exact solution
