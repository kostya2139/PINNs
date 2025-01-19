import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Problem parameters
alpha = tf.constant(0.1, dtype=tf.float32)  # Thermal diffusivity

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
u_ic = (2*np.sin(3*np.pi * x_ic)*np.sin(2*np.pi * y_ic)).astype(np.float32)

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

x_test = np.linspace(0, 1, 40).reshape(-1, 1).astype(np.float32)
y_test = np.linspace(0, 1, 40).reshape(-1, 1).astype(np.float32)
t_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)

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
        print(f"and secondloss {np.linalg.norm(model.predict(np.hstack((X_flat, Y_flat, T_flat))) - 2*np.sin(3*np.pi * X_flat)*np.sin(2*np.pi * Y_flat)*np.exp(-13*np.pi**2 * alpha *T_flat))}")
# Visualize the results


u_pred = model.predict(np.hstack((X_flat, Y_flat, T_flat)))
U_pred = u_pred.reshape(40, 40, 100)


print(np.linalg.norm(u_pred - 2*np.sin(3*np.pi * X_flat)*np.sin(2*np.pi * Y_flat)*np.exp(-13*np.pi**2 * alpha *T_flat)))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X[:,:,0], Y[:,:,0], U_pred[:,:,0], cmap="viridis", edgecolor="none")


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y, t)")
ax.set_zlim(-1, 1)

def update(frame):
    global surf
    surf.remove()  # Remove the old surface
    surf = ax.plot_surface(X[:,:,0], Y[:,:,0], U_pred[:,:,frame], cmap="viridis", edgecolor="none")
    ax.set_title(f"Time: t = {frame/100.0:.2f}")

anim = animation.FuncAnimation(fig, update, frames=100, interval=100)

# Display the animation
plt.show()

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
