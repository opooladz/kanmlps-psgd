import matplotlib.pyplot as plt
loaded_data = np.load('data.npz', allow_pickle=True)

# Extract the data back into the variable names
mean_losses = loaded_data['mean_losses'].item()
std_losses = loaded_data['std_losses'].item()
mean_step_times = loaded_data['mean_step_times'].item()
std_step_times = loaded_data['std_step_times'].item()
mean_memory_usage = loaded_data['mean_memory_usage'].item()
std_memory_usage = loaded_data['std_memory_usage'].item()
# Training parameters
steps = 5000
interval = 100  # Save the model output every `interval` epochs

# Number of datasets to sample
num_datasets = 2
lr0s = {"Simple": 0.2,
        "Expanding": 0.1,
        "Learned Act": 0.15,
        "Reglu": 0.01,
        "KAN": 0.05,
        "Simple_adam": 0.2,
        "Expanding_adam": 0.1,
        "Learned Act_adam": 0.15,
        "Reglu_adam": 0.01,
        "KAN_adam": 0.05,

        }

from scipy.ndimage import uniform_filter1d


# Define the window size for the moving average
window_size = 5

# Apply the uniform filter to smooth the mean values

# Plotting the loss history with fill_between
fig, ax = plt.subplots(figsize=(12, 8))  # Increase the width and height as needed
x_values = range(0, steps, interval)
for name in lr0s.keys():
    mean = mean_losses[name]
    mean = uniform_filter1d(mean, size=window_size)    
    std = std_losses[name]
    std = uniform_filter1d(std, size=window_size)    

    if 'adam' in name.lower():
        linestyle = '--'  # Dotted line
    else:
        linestyle = '-'  # Solid line
    if 'adam' in name.lower():
      ax.plot(x_values, mean, label=name, linestyle=linestyle)
    else:
      ax.plot(x_values, mean, label=name + '_PSGD', linestyle=linestyle)

    ax.fill_between(x_values, mean - std, mean + std, alpha=0.3)

ax.set_ylim(0, 0.02)

ax.set_xlabel('Steps')
ax.set_ylabel('Loss')
# Place the legend outside the plot area
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Loss after x steps')
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["right"].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")
plt.tight_layout()
# Save the plot to a file
# plt.savefig("loss_comparison_with_variance.png", facecolor=fig.get_facecolor(), dpi=300)
print("Plot saved to loss_comparison_with_variance.png")
plt.show()
