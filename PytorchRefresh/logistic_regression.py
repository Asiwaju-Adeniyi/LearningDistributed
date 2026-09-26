#1) Design model (input, output size, forward pass) 
#2) Construct loss and optimizer
#3) Training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

import torch 
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 