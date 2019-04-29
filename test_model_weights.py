import torch
from old_model import Model
import os

path = os.path.join('model', 'epoch_last.model')
model = Model(751)
print(model.model.conv1.weight)
print(model.classifier.weight)

model.load_state_dict(torch.load(path))
print(model.model.conv1.weight)
print(model.classifier.weight)