from PIL     import Image

from abc     import ABCMeta, abstractmethod

import glob
import math
import numpy as np
import os

class SegmentationDataProvider(metaclass=ABCMeta):
    def __init__(self, batchSize, channels, classNumber):
        self.batchSize   = batchSize
        self.channels    = channels
        self.classNumber = classNumber
    
    def __call__(self):
        data, label = self._LoadDataAndLabel()
        
        width  = data.shape[2]
        height = data.shape[1]
        
        x = np.zeros((self.batchSize, height, width, self.channels), dtype=np.float32)
        y = np.zeros((self.batchSize, height, width, self.classNumber), dtype=np.float32)
        
        x[0] = data
        y[0] = label
        
        for index in range(1, self.batchSize):
            data, label = self._LoadDataAndLabel()
            
            x[index] = data
            y[index] = label
        
        return x, y
    
    def _LoadDataAndLabel(self):
        data, label = self._GetNextDataAndLabel()
        
        data  = self._ProcessData(data)
        label = self._ProcessLabel(label)
        
        data, label = self._PreprocessDataAndLabel(data, label)
        
        width  = data.shape[1]
        height = data.shape[0]
        
        return data.reshape(1, height, width, self.channels), label.reshape(1, height, width, self.classNumber)
    
    def _ProcessData(self, data):
        data  = data.copy()
        data -= np.amin(data)
        data /= np.amax(data)
        
        return data
    
    def _ProcessLabel(self, label):
        return label
    
    @abstractmethod
    def _GetNextDataAndLabel(self):
        pass
    
    @abstractmethod
    def _PreprocessDataAndLabel(self, data, label):
        pass

class UNetDataProvider(SegmentationDataProvider):
    def __init__(self, batchSize, channels, classNumber, path, labelSuffix='-label', shuffle=True):
        super().__init__(batchSize, channels, classNumber)
        
        self.labelSuffix = labelSuffix
        self.shuffle     = shuffle
        
        self.patchNames = os.listdir(path)
        self.patchData  = {}
        
        self.patchSequence        = []
        self.patchSequenceIndex   = 0
        self.patchSequenceCounter = -1
        
        for patchName in self.patchNames:
            fileNames  = glob.glob(f'{path}/{patchName}/*.npy')
            fileNames  = [fileName for fileName in fileNames if not self.labelSuffix in fileName]
            fileNumber = len(fileNames)
            
            self.patchData[patchName] = {
                'file-names': [],
                'index': -1
            }
            
            self.patchData[patchName]['file-names'] = fileNames
            self.patchSequence.extend([patchName] * math.ceil(fileNumber / self.batchSize))
            
            if self.shuffle == True:
                np.random.shuffle(self.patchData[patchName]['file-names'])
        
        if self.shuffle == True:
            np.random.shuffle(self.patchSequence)
    
    def _RotateData(self):
        self.patchSequenceCounter += 1
        
        if self.patchSequenceCounter >= self.batchSize:
            self.patchSequenceIndex   += 1
            self.patchSequenceCounter  = 0
            
            if self.patchSequenceIndex >= len(self.patchSequence):
                self.patchSequenceIndex = 0
                
                if self.shuffle == True:
                    np.random.shuffle(self.patchSequence)
        
        patchName                           = self.patchSequence[self.patchSequenceIndex]
        self.patchData[patchName]['index'] += 1
        
        if self.patchData[patchName]['index'] >= len(self.patchData[patchName]['file-names']):
            self.patchData[patchName]['index'] = 0
            
            if self.shuffle == True:
                np.random.shuffle(self.patchData[patchName]['file-names'])
        
        return self.patchData[patchName]['file-names'][self.patchData[patchName]['index']]
    
    def _GetNextDataAndLabel(self):
        dataName  = self._RotateData()
        labelName = dataName.replace('.npy', f'{self.labelSuffix}.npy')
        
        data  = np.load(dataName).astype(np.float32)
        label = np.load(labelName)
        
        return data, label
    
    def _PreprocessDataAndLabel(self, data, label):
        return data, label

class ImageDataProvider(SegmentationDataProvider):
    def __init__(self, batchSize, channels, classNumber, dataPath, labelPath, shuffle=True):
        super().__init__(batchSize, channels, classNumber)
        
        self.shuffle = shuffle
        
        self.dataNames  = glob.glob(f'{dataPath}/*.*')
        self.labelNames = glob.glob(f'{labelPath}/*.*')
        
        self.fileSequence = np.arange(len(self.dataNames))
        self.fileIndex    = -1
        
        if self.shuffle == True:
            np.random.shuffle(self.fileSequence)
    
    def _RotateData(self):
        self.fileIndex += 1
        
        if self.fileIndex >= len(self.fileSequence):
            self.fileIndex = 0
            
            if self.shuffle == True:
                np.random.shuffle(self.fileSequence)
        
        return self.fileSequence[self.fileIndex]
    
    def _GetNextDataAndLabel(self):
        currentFileIndex = self._RotateData()
        
        dataName  = self.dataNames[currentFileIndex]
        labelName = self.labelNames[currentFileIndex]
        
        data  = np.array(Image.open(dataName), dtype=np.float32)
        label = np.array(Image.open(labelName), dtype=np.float32)
        
        return data, label
    
    def _PreprocessDataAndLabel(self, data, label):
        return data, label