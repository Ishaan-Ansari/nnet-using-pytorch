```python
# create a neural network with 2 hidden layers
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        # called inside the child class to inherit the modules of parent class (Here... torch.nn.Module)
        super().__init__()
        
        # Initialize the Nnet Layers using Sequential
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            
            torch.nn.Linear(30, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs),
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        """
        This is our forward pass
        """
        logits = self.layers(x)
        return logits
    
model = NeuralNetwork(50, 3)

# count number of learnable params
num_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(num_params)
print(model.layers[0].weight)

# print(model)
```