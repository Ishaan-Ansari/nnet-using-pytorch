## Setting up Data loaders

```python
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])
```

### We create custom dataset classs

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        """
        This could be file path, database conneectors, and so on.
        """
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """
        Define instructions for returning exactly one instance from the dataset via an index.
        """
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        """
        Here we use `shape` of a tensor to return number for our dataset
        """
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)
```

```python
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x,y)
```

### The result of will look something like this
```json
Batch 1: tensor([[ 2.3000, -1.1000],
        [-0.9000,  2.9000]]) tensor([1, 0])
Batch 2: tensor([[-1.2000,  3.1000],
        [-0.5000,  2.6000]]) tensor([0, 0])
Batch 3: tensor([[ 2.7000, -1.5000]]) tensor([1])
```