import pickle
import sys
import torch.nn as nn
import torch.nn.functional as F
from data.onnx.mnist.model import Net

model = Net()
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)