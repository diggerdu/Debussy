import time
import torch.backends.cudnn as cudnn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
#from util.visualizer import Visualizer
import sys
import numpy as np

cudnn.benchmark = True

opt = TrainOptions().parse()

model = create_model(opt)


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# visualizer = Visualizer(opt)

total_steps = 0
embark_time = time.time()

startEpoch = int(opt.which_epoch) if opt.continue_train else 0

for epoch in range(startEpoch+1, opt.niter + 1):
    epoch_start_time = time.time()
    errorSum = 0.
    accSum = 0.
    counter = 0.
    dataCount = 0
    for i, data in enumerate(dataset):
        total_steps += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
        label = data['Label'].numpy()
        currentStats = model.get_current_errors()
        error = currentStats['G_LOSS'][0]
        predLogits = currentStats['Logits']
        predLabel = np.argmax(predLogits, axis=1)
        accSum += np.sum(np.array(predLabel) == label)
        errorSum += error
        counter += 1
        dataCount += max(label.shape)


    t = time.time() - epoch_start_time
    print(accSum)
    print('epoch ', epoch, ', current error is ', errorSum / counter, ', current acc is ', accSum / dataCount, ' cost time is ', t)
    if time.time() - embark_time > 60 * 2:
        model.save('latest')
        model.save('epoch{}'.format(epoch))
        embark_time = time.time()

