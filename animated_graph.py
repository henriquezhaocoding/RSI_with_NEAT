import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Sample data generation within a while loop
num_items = 10
max_iterations = 100

# Initialize an empty list for each item's values
item_values = [[] for _ in range(num_items)]

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Set the x-axis limits to be constant
ax.set_xlim(0, max_iterations)

# Set the y-axis limits between -1 and 1
ax.set_ylim(-2, 2)

# Initialize x-axis data (constant throughout the plot)
x_data = np.arange(max_iterations)

# Plot initial empty lines for each item
lines = [ax.plot([], [], label=f'Item {i+1}')[0] for i in range(num_items)]
ax.set_title('Dynamic Line Graph')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value')

# Function to update the plot for each iteration
def update(frame):
    # Generate values for each item (replace this with your logic)
    values = [2 * np.random.rand() - 1 for _ in range(num_items)]  # Values between -1 and 1

    # Update the values for each item in the list
    for i, line in enumerate(lines):
        item_values[i].append(values[i])
        line.set_data(x_data[:len(item_values[i])], item_values[i])

    return lines

# Create an animation and assign it to a variable
ani = animation.FuncAnimation(fig, update, frames=max_iterations, repeat=False)

# Display the animation
plt.show()
