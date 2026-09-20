## Training Loop

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
```

### Code below

```python
import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels) # Loss Function

        optimizer.zero_grad()
        loss.backward()
        optmizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

model.eval()
```

### Terminal will look something like this

```
Epoch: 001/003 | Batch 000/002 | Train/Val Loss: 0.75
Epoch: 001/003 | Batch 001/002 | Train/Val Loss: 0.65
Epoch: 002/003 | Batch 000/002 | Train/Val Loss: 0.44
Epoch: 002/003 | Batch 001/002 | Train/Val Loss: 0.13
Epoch: 003/003 | Batch 000/002 | Train/Val Loss: 0.03
Epoch: 003/003 | Batch 001/002 | Train/Val Loss: 0.00
```

> [!NOTE]
> **Preventing undesired gradient accumulation** It is important to include ```opitmizer.zero_grad()``` call in each update round to reset the gradients to zero. Otheriwise the gradient will accumulate, which may be undisired.

After we trained the model, we can use it to make predictions, as shown below:

```python
model.eval()

with torch.no_grad():
    outputs = model(X_train)

print(outputs)
```

