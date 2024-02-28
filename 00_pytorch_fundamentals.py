import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tensor = torch.tensor([[[1,2,3],[2,1,3], [3,2,1]]])
print(tensor)
print(tensor.dtype)
print(tensor.device)
print(tensor.layout)
print(tensor.shape)
print(tensor.size())
print(tensor.numel())
print(tensor[0])
print(tensor.ndim)