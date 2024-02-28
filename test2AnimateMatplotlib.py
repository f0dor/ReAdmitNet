import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 20000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

t = 0
learning_rate = 0.2e-6

# Create a figure and axis for the animation
fig, ax = plt.subplots()
ax.plot(x, y, 'b')

# Initialize empty plot line for the predicted values
line, = ax.plot(x, np.zeros_like(x), 'r')
ax.set_title(f'Iteration: {t}, Loss: 0.0')

def animate(i):
    global a, b, c, d, t

    t += 1
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)
        if t > 100000:
            print(f'y = {a} + {b} x + {c} x^2 + {d} x^3')

    # Backpropagation to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    line.set_ydata(y_pred)
    ax.set_title(f'Iteration: {t}, Loss: {loss:.4f}')
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=200, interval=0)
plt.show()

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')