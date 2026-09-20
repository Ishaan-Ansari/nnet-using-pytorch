### 5. What is the purpose of the .grad attribute in PyTorch Tensors?
- For tracking gradients during backprop.

```python
import torch

X = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = 2*x + y

# gradients
z.backward() # Triggers gradient computation for z with respect to x and y

print(x.grad)
print(y.grad)

# Code effieciency example
with torch.no_grad():
    a = x*2
    print(a.requires_grad)
```

### 6. Explain what CUDA is and how it relates to PyTorch.
CUDA stands for **compute unified device architechture** that is used to leverage the GPU accelaration

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    print("GPU not available)
```

### 8. Describe the steps for creating a neural network model in PyTorch.
- Architecture design 
- Data prep
- Model construction
- Loss and optimizer selection
- Training loop 
- Eval pipeline
- Inference

#### Basic Neural Netwrk code module
```python 
import torch
from torch.utils.data import Dataset, DataLoaders

## Here we need to define our number of inputs or num outputs or num of layersor optimizers to be used

## Next is how we can fetch the data 
class NnetDataset(Dataset):
    def __init__(self, X, y):
        """
        Here you need to define the path(usually) how do you expect to fetch data
        """
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """
        Single instance of the data
        """
        one_X = self.features[index]
        one_y = self.labels[index]

        return one_X, one_y

    def __len__():
        """
        Size of the data set (or in a way number of rows in your dataset)
        """
        

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(


        )

```
