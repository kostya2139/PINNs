import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Problem parameters
alpha = tf.constant(0.05, dtype=tf.float32)  # Thermal diffusivity

# Neural network architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3,)),  # Input: (x, y, t)
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(1)  # Output: u(x, t)
    ])
    return model

# Physics-Informed Neural Network (PINN)
model = create_model()

# Compute derivatives using TensorFlow's GradientTape
def pde_loss(x, y, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        u = model(tf.concat([x, y, t], axis=1))
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        u_xx = tape.gradient(u_x, x)
        u_y = tape.gradient(u, y)
        u_yy = tape.gradient(u_y, y)
    del tape

    # Residual of the heat equation
    residual = u_t - alpha * (u_xx + u_yy)
    return tf.reduce_mean(tf.square(residual))

# Loss for initial and boundary conditions
def ic_loss(x_ic, y_ic, t_ic, u_ic):
    u_pred = model(tf.concat([x_ic,  y_ic, t_ic], axis=1))
    return tf.reduce_mean(tf.square(u_pred - u_ic))

def bc_loss(x_bc, y_ic, t_bc):
    u_pred = model(tf.concat([x_bc, y_ic, t_bc], axis=1))
    return tf.reduce_mean(tf.square(u_pred))

# Generate training data
N_collocation = 10000  # Collocation points for the PDE
N_ic = 100  # Initial condition points
N_bc = 100  # Boundary condition points

# Collocation points (x, t) in the domain [0, 1] x [0, 1]
x_collocation = np.random.uniform(0, 1, (N_collocation, 1)).astype(np.float32)
y_collocation = np.random.uniform(0, 1, (N_collocation, 1)).astype(np.float32)
t_collocation = np.random.uniform(0, 1, (N_collocation, 1)).astype(np.float32)

# Initial condition: u(x, 0) = sin(pi * x)
x_ic = np.random.uniform(0, 1, (N_ic, 1)).astype(np.float32)
y_ic = np.random.uniform(0, 1, (N_ic, 1)).astype(np.float32)
t_ic = np.zeros((N_ic, 1), dtype=np.float32)
a = 2
k_x = 2
k_y = 2
u_ic = (a*np.sin(k_x*np.pi * x_ic)*np.sin(k_y*np.pi * y_ic)).astype(np.float32)

# Boundary conditions: u(0, t) = 0, u(1, t) = 0
t_bc = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
x_bc_0 = np.zeros((N_bc, 1), dtype=np.float32)
x_bc_1 = np.ones((N_bc, 1), dtype=np.float32)
x_bc_2 = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
x_bc_3 = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
y_bc_0 = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
y_bc_1 = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
y_bc_2 = np.zeros((N_bc, 1), dtype=np.float32)
y_bc_3 = np.ones((N_bc, 1), dtype=np.float32)

# Training the PINN
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(x_collocation, y_collocation, t_collocation, x_ic, t_ic, u_ic, x_bc_0, y_bc_0, x_bc_1, y_bc_1, x_bc_2, y_bc_2, x_bc_3, y_bc_3, t_bc):
    with tf.GradientTape() as tape:
        # Compute losses
        loss_pde = pde_loss(x_collocation, y_collocation, t_collocation)

        loss_ic = ic_loss(x_ic, y_ic, t_ic, u_ic)
        loss_bc_0 = bc_loss(x_bc_0, y_bc_0, t_bc)
        loss_bc_1 = bc_loss(x_bc_1, y_bc_1, t_bc)
        loss_bc_2 = bc_loss(x_bc_2, y_bc_2, t_bc)
        loss_bc_3 = bc_loss(x_bc_3, y_bc_3, t_bc)

        # Total loss
        total_loss = loss_pde + loss_ic + loss_bc_0 + loss_bc_1 + loss_bc_2 + loss_bc_3

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

n_x_t = 40
n_y_t = 40
n_t_t = 100
x_test = np.linspace(0, 1, n_x_t).reshape(-1, 1).astype(np.float32)
y_test = np.linspace(0, 1, n_y_t).reshape(-1, 1).astype(np.float32)
t_test = np.linspace(0, 1, n_t_t).reshape(-1, 1).astype(np.float32)

X, Y, T = np.meshgrid(x_test, y_test, t_test)
X_flat = X.flatten().reshape(-1, 1).astype(np.float32)
Y_flat = Y.flatten().reshape(-1, 1).astype(np.float32)
T_flat = T.flatten().reshape(-1, 1).astype(np.float32)

# Training loop
epochs = 2000
for epoch in range(epochs):
    loss = train_step(x_collocation, y_collocation, t_collocation, x_ic, t_ic, u_ic, x_bc_0, y_bc_0, x_bc_1, y_bc_1, x_bc_2, y_bc_2, x_bc_3, y_bc_3, t_bc)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4e} ")
    if epoch % 500 == 0:
        print(f"and secondloss {np.linalg.norm((model.predict(np.hstack((X_flat, Y_flat, T_flat))) - a*np.sin(k_x*np.pi * X_flat)*np.sin(k_y*np.pi * Y_flat)*np.exp(-(k_x**2 + k_y**2)*np.pi**2 * alpha *T_flat)).reshape(-1))}")
# Visualize the results


u_pred = model.predict(np.hstack((X_flat, Y_flat, T_flat)))
U_pred = u_pred.reshape(n_x_t, n_y_t, n_t_t)


##print(np.linalg.norm((u_pred - a*np.sin(k_x*np.pi * X_flat)*np.sin(k_y*np.pi * Y_flat)*np.exp(-(k_x**2 + k_y**2)*np.pi**2 * alpha *T_flat)).reshape(-1)) / np.linalg.norm((a*np.sin(k_x*np.pi * X_flat)*np.sin(k_y*np.pi * Y_flat)*np.exp(-(k_x**2 + k_y**2)*np.pi**2 * alpha *T_flat)).reshape(-1)))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X[:,:,0], Y[:,:,0], U_pred[:,:,0], cmap="viridis", edgecolor="none")
cbar = fig.colorbar(surf, ax=ax, label="u(x, y, t)")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y, t)")
ax.set_zlim(-1, 1)

def update(frame):
    global surf, n_t_t
    surf.remove()  # Remove the old surface
    surf = ax.plot_surface(X[:,:,0], Y[:,:,0], U_pred[:,:,frame], cmap="viridis", edgecolor="none")
    ax.set_title(f"Time: t = {frame*1.0/n_t_t:.2f}")

import os
os.makedirs("frames", exist_ok=True)

# Generate and save frames
for i in range(n_t_t): 
    update(i)
    plt.savefig(f"frames/frame_{i:03d}.png")  # Save as frame_000.png, frame_001.png, etc.

from PIL import Image
import os

# Directory containing the frames
frames_dir = "frames"

# Output GIF filename
output_gif = "animation.gif"

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

# Display the animation
#plt.show()

# plt.figure(figsize=(8, 6))


# fig, ax = plt.subplots()
# plt.xlabel("x")
# plt.ylabel("t")
# plt.title("Solution of the Heat Equation via PINN")
# # Initialize the plot
# contour = ax.contourf(X[:,:,0], Y[:,:,0], U_pred[:,:,0], levels=50, cmap="viridis")
# # Update function for animation
# def update(frame):
#     ax.clear()
#     ax.contourf(X[:,:,0], Y[:,:,0], U_pred[:,:,frame], levels=50, cmap="viridis")
#     return contour.collections

# # Create the animation
# ani =animation.FuncAnimation(fig, update, frames=100, interval=100)
# #plt.contourf(X, T, U_pred, levels=50, cmap="viridis")

# plt.show()
