from NeuralNetwork        import Configuration
from NeuralNetwork.Logger import logger

from PIL                  import Image

from Setting              import DATA_PATH, LABEL_PATH, SEED, TRAINING_DATA_PATH, VALIDATION_DATA_PATH

from Utility              import CutPatchByGrid

import glob
import numpy              as np
import os
import random

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

Configuration.LoadConfiguration('.')

dataPathes  = glob.glob(f'{DATA_PATH}/*')
labelPathes = glob.glob(f'{LABEL_PATH}/*')

totalFrequency = np.zeros(Configuration.classNumber, dtype=np.float32)

for index in range(len(dataPathes)):
    imageName = dataPathes[index].split('\\')[-1]
    
    logger.info(f'Making Patches From {imageName}')
    
    data  = np.array(Image.open(dataPathes[index]))
    label = np.array(Image.open(labelPathes[index]))
    
    frequency       = CutPatchByGrid(data, label, TRAINING_DATA_PATH, VALIDATION_DATA_PATH)
    totalFrequency += frequency

totalFrequency = 1.0 / totalFrequency
classWeights   = totalFrequency / np.sum(totalFrequency)
classWeights   = ', '.join(map(str, classWeights))

logger.info(f'Class Weights: {classWeights}')