import torch
import torch.nn as nn

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(100, 200)
        self.layer2 = nn.Linear(200, 500)
        self.layer3 = nn.Linear(500, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Create an instance of the neural network and move it to the GPU
print(f'Are u sure calculate using {device}?')
maualat = input('Yes/No:')
if maualat=='Yes':
    pass
elif maualat=='No':
    device = input('cuda/cpu:')

model = SimpleNN().to(device)

# Create some dummy input data and move it to the GPU
input_data = torch.randn(1, 100).to(device)
print(input_data)

# Perform a forward pass
output = model(input_data)

print('Output:', output)
