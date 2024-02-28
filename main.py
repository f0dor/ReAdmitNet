import torch

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x+y)
z = torch.add(x,y)
print(z)