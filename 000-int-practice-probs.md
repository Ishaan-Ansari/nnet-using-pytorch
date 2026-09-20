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
        return self.labels.shape[0]

train_ds = NnetDataset(X_train, y_train)
test_ds = NnetDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, num_outputs),
            torch.nn.ReLU()
        )


    def forward(self, x):
        logits = self.layers(x)
        return logits


model = NeuralNetwork(2, 3)

### Loss and Optimizer selection
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)

## Training Loop
num_epoch = 3
for epoch in range(num_epoch):
    model.train()
    for batch, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = criterion(logits, labels)

        optimizer.zero_grad()   # zero_grad is used to reset the gradients of all model parameters to zero
        loss.backward()         # compute loss
        optimizer.step()        # update weights

        ## Add some logging in order watch validation at each step

model.eval()


```
