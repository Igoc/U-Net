from NeuralNetwork import Configuration

from Setting       import PATCH_SPLIT_PROBABILITY

import glob
import numpy       as np
import os

def EncodeLabelAsOneHot(label):
    output     = np.zeros((label.shape[0], label.shape[1], Configuration.classNumber), dtype=np.float32)
    pixelCount = np.zeros(Configuration.classNumber, dtype=np.float32)
    classCount = 0
    
    for classColor in Configuration.classColors:
        red   = label[:, :, 0] == classColor[0]
        green = label[:, :, 1] == classColor[1]
        blue  = label[:, :, 2] == classColor[2]
        
        index = red & green & blue
        
        output[index, classCount] = 1
        pixelCount[classCount]    = np.count_nonzero(index)
        
        classCount += 1
    
    return output, pixelCount

def ConvertOneHotToColor(label):
    output      = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    classColors = np.array(Configuration.classColors)
    
    for classIndex in range(Configuration.classNumber):
        index            = label == classIndex
        output[index, :] = classColors[classIndex, :]
    
    return output

def CutPatchByGrid(data, label, trainingDataPath='./Data/Training', validationDataPath='./Data/Validation'):
    frequency = np.zeros(Configuration.classNumber, dtype=np.float32)
    
    for index in range(len(Configuration.patchSizes)):
        patchSize   = Configuration.patchSizes[index]
        patchStride = Configuration.patchStrides[index]
        
        trainingPatchPath   = f'{trainingDataPath}/{patchSize}'
        validationPatchPath = f'{validationDataPath}/{patchSize}'
        
        if os.path.isdir(trainingPatchPath) == False:
            os.makedirs(trainingPatchPath)
        
        if os.path.isdir(validationPatchPath) == False:
            os.makedirs(validationPatchPath)
        
        trainingIndex   = len(glob.glob(f'{trainingPatchPath}/*.npy')) // 2
        validationIndex = len(glob.glob(f'{validationPatchPath}/*.npy')) // 2
        
        width  = data.shape[1]
        height = data.shape[0]
        
        for iy in range(0, height - patchSize, patchStride):
            for ix in range(0, width - patchSize, patchStride):
                patchData  = data[iy:iy + patchSize, ix:ix + patchSize]
                patchLabel = label[iy:iy + patchSize, ix:ix + patchSize]
                
                patchLabel, pixelCount = EncodeLabelAsOneHot(patchLabel)
                
                if np.random.uniform() <= PATCH_SPLIT_PROBABILITY:
                    np.save(f'{trainingPatchPath}/{trainingIndex:07}', patchData)
                    np.save(f'{trainingPatchPath}/{trainingIndex:07}-label', patchLabel)
                    
                    trainingIndex += 1
                    frequency     += pixelCount
                else:
                    np.save(f'{validationPatchPath}/{validationIndex:07}', patchData)
                    np.save(f'{validationPatchPath}/{validationIndex:07}-label', patchLabel)
                    
                    validationIndex += 1
    
    return frequency