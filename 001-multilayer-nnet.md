```python
# create a neural network with 2 hidden layers
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs),
        )
        
    def forward(self, x):
        logits = self.layers(x)
        return logits
    
    
# incase if we want to keep the weights intialization reproducable we can do this
torch.manual_seed(123)
model = NeuralNetwork(50, 3)

# now we can also check the results of forward pass
# note that our network expects 50-dimensional feature vectors
x = torch.rand((1, 50))
out = model(x)
print(out)

# count number of learnable params
num_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(num_params)
print(model.layers[0].weight)

# print(model)
```