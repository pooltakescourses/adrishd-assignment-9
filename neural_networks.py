import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import tqdm
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        # To store intermediate activations for visualization
        self.hidden_activations = None
        self.gradients = None

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

    def forward(self, X):
        # Forward pass
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)  # Hidden layer activation
        self.Z2 = self.A1 @ self.W2 + self.b2
        out = np.tanh(self.Z2)  # Output layer activation
        self.hidden_activations = self.A1
        return out

    def backward(self, X, y):
        # Backward pass
        m = X.shape[0]
        # Gradient of loss with respect to output
        dZ2 = 2 * (self.forward(X) - y) * (1 - np.tanh(self.Z2) ** 2)
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.gradients = [dW1, dW2]

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, bar):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    bar.update(1)

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # --- Hidden Layer Visualization ---
    hidden_features = mlp.hidden_activations
    
    # Create a grid in input space for manifold visualization
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
    grid_input = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Transform grid points through the network
    grid_hidden = mlp.activation(np.dot(grid_input, mlp.W1) + mlp.b1)
    
    # Create decision boundary plane in hidden space
    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    zz = (-mlp.W2[0,0] * xx - mlp.W2[1,0] * yy - mlp.b2[0]) / mlp.W2[2,0]
    
    # Plot decision boundary plane
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.2, color='tan')
    
    # Plot the manifold surface
    ax_hidden.plot_surface(
        grid_hidden.reshape(20, 20, 3)[:, :, 0],
        grid_hidden.reshape(20, 20, 3)[:, :, 1],
        grid_hidden.reshape(20, 20, 3)[:, :, 2],
        alpha=0.3,
        color='blue'
    )


    # Plot the transformed data points
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        s=50
    )

    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)
    # --- Input Space Visualization ---
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid) > 0
    predictions = predictions.reshape(xx.shape)
    
    ax_input.contourf(xx, yy, predictions, levels=1, colors=['blue', 'red'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)

    # --- Gradient Visualization ---
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.axis('off')
    
    layers = [X.shape[1], mlp.W1.shape[1], mlp.W2.shape[1]]
    positions = [np.linspace(0, 1, n) for n in layers]
    positions_x = [np.full(n, i) for i, n in enumerate(layers)]

    # Draw neurons
    for i, layer in enumerate(layers):
        ax_gradient.scatter(positions_x[i], positions[i], s=300, c='blue')

    # Draw connections
    for i, (layer_1, layer_2) in enumerate(zip(layers[:-1], layers[1:])):
        weights = mlp.gradients[i]
        weight_magnitudes = np.abs(weights)
        
        for j in range(layer_1):
            for k in range(layer_2):
                weight_strength = weight_magnitudes[j, k]
                thickness = np.clip(100.0 * weight_strength, 0, 5.0)
                alpha = np.clip(weight_strength, 0, 1.0)
                ax_gradient.plot(
                    [positions_x[i][j], positions_x[i+1][k]],
                    [positions[i][j], positions[i+1][k]],
                    color='purple',
                    lw=thickness,
                    alpha=alpha
                )
    # --- Rotate the Hidden Space Plot Slowly ---
    ax_hidden.view_init(elev=30., azim=frame * 2)  # Slowly rotate the plot as the animation progresses

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    bar = tqdm.tqdm(range(step_num // 10 + 1))

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y, bar=bar), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
