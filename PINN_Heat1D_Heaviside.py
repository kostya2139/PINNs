import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools

#Constant
alpha = 0.04

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
def loss(x, t, u):
    u_pred = model(tf.concat([x, t], axis=1))
    return tf.reduce_mean(tf.square(u_pred - u))

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
u_ic = np.array([1 if x_ic[i][0] >=0.5 else 0 for i in range(N_ic)]).reshape(N_ic, 1).astype(np.float32)  # Gaussian initial condition

# Boundary conditions: u(0, t) = 0, u(1, t) = 0
t_bc = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
x_bc_0 = np.zeros((N_bc, 1), dtype=np.float32)
x_bc_1 = np.ones((N_bc, 1), dtype=np.float32)
u_bc_0 = np.zeros((N_bc, 1), dtype=np.float32)
u_bc_1 = np.ones((N_bc, 1), dtype=np.float32)

# Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def train_step(x_collocation, t_collocation, x_ic, t_ic, u_ic, x_bc_0, x_bc_1, t_bc, u_bc_0, u_bc_1):
    with tf.GradientTape() as tape:
        loss_pde = pde_loss(x_collocation, t_collocation)

        loss_ic = loss(x_ic, t_ic, u_ic)
        loss_bc_0 = loss(x_bc_0, t_bc, u_bc_0)
        loss_bc_1 = loss(x_bc_1, t_bc, u_bc_1)

        total_loss = loss_pde + loss_ic + loss_bc_0 + loss_bc_1

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return (total_loss, loss_ic, loss_pde, loss_bc_0, loss_bc_1)

x_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
t_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)

X, T = np.meshgrid(x_test, t_test)
X_flat = X.flatten().reshape(-1, 1).astype(np.float32)
T_flat = T.flatten().reshape(-1, 1).astype(np.float32)


# Training loop (same as before)
epochs = 12000
for epoch in range(epochs):
    loss = train_step(x_collocation, t_collocation, x_ic, t_ic, u_ic, x_bc_0, x_bc_1, t_bc, u_bc_0, u_bc_1)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}")
        for data in loss:
            print(f"Loss:{data.numpy():.4e}")
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

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.set_title("Solution of the Heat Equation via PINN")
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_ylim(-0.2, 1.2)
line, = ax.plot(X[0, :], U_pred[0, :])

def update(frame):
    global line
    line.set_ydata(U_pred[frame, :])
    ax.set_title(f"Time: t = {frame*1.0/100:.2f}")


import os
os.makedirs("frames", exist_ok=True)

# Generate and save frames
for i in range(100): 
    update(i)
    plt.savefig(f"frames/frame_{i:03d}.png")  # Save as frame_000.png, frame_001.png, etc.

from PIL import Image
# Directory containing the frames
frames_dir = "frames"

# Output GIF filename
output_gif = "animation_Heaviside.gif"

# Get all frame filenames sorted numerically
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".png")],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

# Load frames into a list
frames = [Image.open(os.path.join(frames_dir, frame)) for frame in frame_files]

# Save as GIF
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],  # Add the rest of the frames
    duration=100,  # Duration of each frame in milliseconds (e.g., 100ms = 10fps)
    loop=0  # Loop forever (use 1 for no loop)
)
#print(np.linalg.norm(u_pred - np.exp(-((X_flat - 0.5)**2) / 0.1)))  # Error between predicted and exact solution
