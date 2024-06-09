import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(200000, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# Create random input data and labels
input_data = torch.randn(8092, 200000)
labels = torch.randint(0, 10, (8092,))

# Define the device to use (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'

# Instantiate the model and move it to the device
model = SimpleNet().to(device)
input_data = input_data.to(device)
labels = labels.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
start_time = time.time()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

end_time = time.time()

# Calculate the time taken
elapsed_time = end_time - start_time
print(f"Time taken for training on {device}: {elapsed_time} seconds")