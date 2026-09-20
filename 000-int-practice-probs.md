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
