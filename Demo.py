from NeuralNetwork              import Configuration
from NeuralNetwork.DataProvider import ImageDataProvider
from NeuralNetwork.UNet         import UNet

from Setting                    import DATA_PATH, LABEL_PATH, TRAINING_OUTPUT_PATH, VALIDATION_OUTPUT_PATH

from Utility                    import ConvertOneHotToColor

import matplotlib.pyplot        as plt
import numpy                    as np
import os
import tensorflow               as tf

Configuration.LoadConfiguration('.')

if os.path.exists(VALIDATION_OUTPUT_PATH) == False:
    os.makedirs(VALIDATION_OUTPUT_PATH)

networkCostParameter = {
    'class-weights': Configuration.classWeights
}

networkParameters = {
    'layers': Configuration.layerNumber,
    'featuresRoot': Configuration.featuresRoot
}

network      = UNet(3, Configuration.classNumber, 'cross-entropy', networkCostParameter, **networkParameters)
dataProvider = ImageDataProvider(1, 3, 3, DATA_PATH, LABEL_PATH, False)

dataNames = os.listdir(DATA_PATH)

with tf.Session() as session:
    network.LoadModel(session, f'{TRAINING_OUTPUT_PATH}/model.cpkt')
    
    figure, axes = plt.subplots(1, Configuration.classNumber + 2, figsize=(45, 10))
    
    for index in range(len(dataNames)):
        x, _       = dataProvider()
        prediction = network.Predict(session, x)[0]
        
        width  = x.shape[2]
        height = x.shape[1]
        
        axes[0].imshow(x[0], aspect='auto')
        
        output = np.argmax(prediction, axis=2)
        output = ConvertOneHotToColor(output)
        
        for classIndex in range(Configuration.classNumber):
            axes[classIndex + 1].imshow(np.reshape(prediction, (1, height, width, Configuration.classNumber))[0, ..., classIndex], aspect='auto', cmap='jet')
        
        axes[Configuration.classNumber + 1].imshow(output, aspect='auto')
        
        figure.tight_layout()
        figure.savefig(f'{VALIDATION_OUTPUT_PATH}/{dataNames[index]}')
        
        for axe in axes:
            axe.clear()