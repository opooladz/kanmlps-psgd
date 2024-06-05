import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm

torch.set_float32_matmul_precision('high')

# Training parameters
steps = 1000
interval = 10  # Save the model output every `interval` epochs

# Number of datasets to sample
num_datasets = 10

# Function to generate random dataset
def generate_dataset(seed):
    torch.manual_seed(1212343 + seed)
    x = torch.linspace(-2, 2, 500)
    y0 = torch.special.bessel_j0(20 * x)
    y = y0 + .1 * torch.randn(x.shape)
    x_tensor = x.view(-1, 1).float()
    y_tensor = y.view(-1, 1).float()
    return x_tensor.cuda(), y_tensor.cuda(), y0.cuda()

epistemic_loss = 0
for i in range(num_datasets):
    x_tensor, y_tensor, y0 = generate_dataset(seed=i)
    epistemic_loss += nn.MSELoss()(y0.view(-1, 1).float(), y_tensor).item()
epistemic_loss /= num_datasets
print(f"Epistemic loss: {epistemic_loss:.4f}")

# List of network architectures
from models import SimpleNet, ExpansionMLP, LearnedActivationMLP, RegluMLP, Kan, RegluExpandMLP, Mix2MLP

# All architectures are 2-hidden layer MLPs with (1, 100, 100, 1) units
architectures = {
    # Params: 2d^2 + O(d)
    "Simple": SimpleNet(d=100),
    # Params: 4kd^2 + O(d)
    "Expanding": ExpansionMLP(d=100, k=3),
    # Params: 2d^2 + 6dk + O(d)
    "Learned Act": LearnedActivationMLP(d=100, k=3),
    # Params: 4d^2 + O(d)
    "Gated Sine": RegluMLP(d=100, func=torch.sin),
    "Reglu": RegluMLP(d=100, func=torch.relu),
    # Params: 6kd^2 + O(dk)
    "KAN": Kan(d=100, k=3, scale=.5, func=torch.relu),
    # Params: 4kd^2 + O(dk)
    "MoE": Mix2MLP(d=100, k=3, func=torch.relu),
}
architectures = {
        #name: net.cuda()
        name: torch.compile(net.cuda())
        for name, net in architectures.items()
        }

# architectures = {
#     "Kan s=2": Kan(d=100, k=10, scale=2),
#     "Kan s=1": Kan(d=100, k=10, scale=1),
#     "Kan s=.5": Kan(d=100, k=10, scale=.5),
#     "Kan s=.2": Kan(d=100, k=10, scale=.2),
# }

# Dictionary to store loss history
loss_history = {name: [] for name in architectures.keys()}

# Training loop for each architecture on each dataset
for dataset_id in range(num_datasets):
    print(f"Training on Dataset {dataset_id+1}")
    x_tensor, y_tensor, _ = generate_dataset(seed=dataset_id)
    for name, net in architectures.items():
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        losses = []
        with tqdm.tqdm(range(steps), desc=f"Training {name} on Dataset {dataset_id+1}") as pbar:
            for step in pbar:
                net.train()
                optimizer.zero_grad()
                outputs = net(x_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                if step % interval == 0:
                    losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})

        loss_history[name].append(losses)

# Compute mean and standard deviation of losses
mean_losses = {name: np.mean(losses, axis=0) for name, losses in loss_history.items()}
std_losses = {name: np.std(losses, axis=0) for name, losses in loss_history.items()}

# Plotting the loss history with fill_between
fig, ax = plt.subplots()
x_values = range(0, steps, interval)
for name in architectures.keys():
    mean = mean_losses[name]
    std = std_losses[name]
    ax.plot(x_values, mean, label=name)
    ax.fill_between(x_values, mean - std, mean + std, alpha=0.3)

# Generate the first dataset to compute Epistemic loss
# ax.axhline(y=epistemic_loss, color='r', linestyle='--', label='Epistemic Loss')

# Determine y-axis limits by ignoring outliers
all_losses = np.concatenate([np.concatenate(losses) for losses in loss_history.values()])
q1, q3 = np.percentile(all_losses, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
ax.set_ylim(lower_bound, upper_bound)

ax.set_xlabel('Steps')
ax.set_ylabel('Loss')
ax.legend()
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

# Save the plot to a file
plt.savefig("loss_comparison_with_variance.png", facecolor=fig.get_facecolor(), dpi=300)
print("Plot saved to loss_comparison_with_variance.png")

plt.show()
