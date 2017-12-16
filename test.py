import time
import torch.backends.cudnn as cudnn
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
#from util.visualizer import Visualizer
import sys
import numpy as np

cudnn.benchmark = True

opt = TestOptions().parse()

model = create_model(opt)


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# visualizer = Visualizer(opt)

total_steps = 0

embark_time = time.time()
for i, data in enumerate(dataset):
    total_steps += 1
    model.set_input(data)
    model.test()
