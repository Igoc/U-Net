from NeuralNetwork              import Configuration
from NeuralNetwork.DataProvider import UNetDataProvider
from NeuralNetwork.Trainer      import Trainer
from NeuralNetwork.UNet         import UNet

from Setting                    import SEED, TRAINING_DATA_PATH, TRAINING_LOG_PATH, TRAINING_OUTPUT_PATH

import numpy                    as np
import os
import random
import tensorflow               as tf

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

Configuration.LoadConfiguration('.')

networkCostParameter = {
    'class-weights': Configuration.classWeights
}

networkParameters = {
    'layers': Configuration.layerNumber,
    'featuresRoot': Configuration.featuresRoot
}

network      = UNet(3, Configuration.classNumber, 'cross-entropy', networkCostParameter, **networkParameters)
dataProvider = UNetDataProvider(Configuration.batchSize, 3, Configuration.classNumber, TRAINING_DATA_PATH)

dataNumber = 0

for patchName in dataProvider.patchData.keys():
    dataNumber += len(dataProvider.patchData[patchName]['file-names'])

trainerParameters = {
    'learning-rate': Configuration.learningRate,
    'decay-steps': dataNumber // Configuration.batchSize,
    'decay-rate': Configuration.learningRateDecay
}

trainer = Trainer(network, 'adam', False, **trainerParameters)
trainer.Train(dataProvider, dataNumber // Configuration.batchSize, TRAINING_OUTPUT_PATH, TRAINING_LOG_PATH, Configuration.epoch, 0.5, 100, True, False)