from collections          import OrderedDict

from NeuralNetwork.Layers import BiasVariable, ConvolutionLayer2D, CropAndConcatTensor, CrossEntropyLoss, DeconvolutionLayer2D, MaxPoolingLayer, PixelWiseSoftmaxActivation, WeightVariable
from NeuralNetwork.Logger import logger

import numpy              as np
import tensorflow         as tf

class UNet(object):
    def __init__(self, channels=3, classNumber=2, cost='cross-entropy', costKwargs={}, **kwargs):
        tf.reset_default_graph()
        
        self.classNumber = classNumber
        self.summaries   = kwargs.get('summaries', True)
        
        self.x               = tf.placeholder(tf.float32, shape=[None, None, None, channels])
        self.y               = tf.placeholder(tf.float32, shape=[None, None, None, classNumber])
        self.keepProbability = tf.placeholder(tf.float32)
        self.classWeights    = tf.placeholder(tf.float32, shape=[classNumber])
        
        self.outputMap, self.variables, self.offset = self._CreateConvolutionalNetwork(self.x, self.keepProbability, channels, classNumber, **kwargs)
        
        self.globalWeights = []
        self.cost          = self._GetCost(self.outputMap, cost, costKwargs)
        
        self.gradientsNode = tf.gradients(self.cost, self.variables)
        self.crossEntropy  = tf.reduce_mean(CrossEntropyLoss(tf.reshape(self.y, [-1, classNumber]), tf.reshape(PixelWiseSoftmaxActivation(self.outputMap), [-1, classNumber])))
        
        self.prediction        = PixelWiseSoftmaxActivation(self.outputMap)
        self.correctPrediction = tf.equal(tf.argmax(self.prediction, 3), tf.argmax(self.y, 3))
        self.accuracy          = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))
    
    def LoadModel(self, session, path):
        session.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        saver.restore(session, path)
        
        logger.info(f'Loading Model From {path}')
    
    def Predict(self, session, x):
        dummyY     = np.empty((x.shape[0], x.shape[1], x.shape[2], self.classNumber))
        prediction = session.run(self.prediction, feed_dict={
            self.x: x,
            self.y: dummyY,
            self.keepProbability: 1.0
        })
        
        return prediction
    
    def SaveModel(self, session, path):
        saver    = tf.train.Saver()
        savePath = saver.save(session, path)
        
        logger.info(f'Saving Model To {savePath}')
        
        return savePath
    
    def _CreateConvolutionalNetwork(self, x, keepProbability, channels, classNumber, layers=3, featuresRoot=16, filterSize=3, poolingSize=2, summaries=True):
        logger.info(f'Layers: {layers}, Features: {featuresRoot}, Filter Size: {filterSize}×{filterSize}, Pooling Size: {poolingSize}×{poolingSize}')
        
        width      = tf.shape(x)[1]
        height     = tf.shape(x)[2]
        inputImage = tf.reshape(x, tf.stack([-1, width, height, channels]))
        size       = 1000
        
        inputNode = inputImage
        inputSize = size
        
        weights          = []
        biases           = []
        convolutions     = []
        poolings         = OrderedDict()
        deconvolutions   = OrderedDict()
        downConvolutions = OrderedDict()
        upConvolutions   = OrderedDict()
        
        for layer in range(0, layers):
            feature = 2 ** layer * featuresRoot
            
            with tf.variable_scope(f'conv{layer}'):
                if layer == 0:
                    weight1 = WeightVariable([filterSize, filterSize, channels, feature], 'w1')
                else:
                    weight1 = WeightVariable([filterSize, filterSize, feature // 2, feature], 'w1')
                
                weight2 = WeightVariable([filterSize, filterSize, feature, feature], 'w2')
                bias1   = BiasVariable([feature])
                bias2   = BiasVariable([feature])
                
                convolution1 = ConvolutionLayer2D(inputNode, weight1)
                convolution2 = ConvolutionLayer2D(tf.nn.relu(convolution1 + bias1), weight2)
                
                if layer == layers - 1 or layer == layers - 2:
                    downConvolutions[layer] = tf.nn.dropout(tf.nn.relu(convolution2 + bias2), keepProbability)
                else:
                    downConvolutions[layer] = tf.nn.relu(convolution2 + bias2)
                
                weights.append((weight1, weight2))
                biases.append((bias1, bias2))
                convolutions.append((convolution1, convolution2))
                
                inputSize -= 4
                
                if layer < layers - 1:
                    poolings[layer]  = MaxPoolingLayer(downConvolutions[layer], poolingSize)
                    inputNode        = poolings[layer]
                    inputSize       /= 2
        
        inputNode = downConvolutions[layers - 1]
        
        for layer in range(layers - 2, -1, -1):
            feature = 2 ** (layer + 1) * featuresRoot
            
            with tf.variable_scope(f'deconv{layer}'):
                deconvolutionWeight = WeightVariable([poolingSize, poolingSize, feature // 2, feature], 'wd')
                deconvolutionBias   = BiasVariable([feature // 2])
                
                deconvolution         = tf.nn.relu(DeconvolutionLayer2D(inputNode, deconvolutionWeight, poolingSize) + deconvolutionBias)
                concatedDeconvolution = CropAndConcatTensor(downConvolutions[layer], deconvolution)
                deconvolutions[layer] = concatedDeconvolution
                
                weight1 = WeightVariable([filterSize, filterSize, feature, feature // 2], 'w1')
                weight2 = WeightVariable([filterSize, filterSize, feature // 2, feature // 2], 'w2')
                bias1   = BiasVariable([feature // 2])
                bias2   = BiasVariable([feature // 2])
                
                convolution1 = ConvolutionLayer2D(concatedDeconvolution, weight1)
                convolution2 = ConvolutionLayer2D(tf.nn.relu(convolution1 + bias1), weight2)
                
                inputNode             = tf.nn.relu(convolution2 + bias2)
                upConvolutions[layer] = inputNode
                
                weights.append((weight1, weight2))
                biases.append((bias1, bias2))
                convolutions.append((convolution1, convolution2))
            
            inputSize *= 2
            inputSize -= 4
        
        weight      = WeightVariable([1, 1, featuresRoot, classNumber], 'outw')
        bias        = BiasVariable([classNumber])
        convolution = ConvolutionLayer2D(inputNode, weight)
        
        outputMap             = tf.nn.relu(convolution + bias)
        upConvolutions['Out'] = outputMap
        
        if summaries == True:
            for index, (convolution1, convolution2) in enumerate(convolutions):
                tf.summary.image(f'Convolution-{index:02}-01', self._GetImageSummary(convolution1))
                tf.summary.image(f'Convolution-{index:02}-02', self._GetImageSummary(convolution2))
            
            for key in poolings.keys():
                tf.summary.image(f'Pooling-{key:02}', self._GetImageSummary(poolings[key]))
            
            for key in deconvolutions.keys():
                tf.summary.image(f'Concated-Deconvolution-{key:02}', self._GetImageSummary(deconvolutions[key]))
            
            for key in downConvolutions.keys():
                tf.summary.histogram(f'Down-Convolution-{key:02}', downConvolutions[key])
            
            for key in upConvolutions.keys():
                tf.summary.histogram(f'Up-Convolution-{key}', upConvolutions[key])
        
        variables = []
        
        for weight1, weight2 in weights:
            variables.append(weight1)
            variables.append(weight2)
        
        for bias1, bias2 in biases:
            variables.append(bias1)
            variables.append(bias2)
        
        return outputMap, variables, int(size - inputSize)
    
    def _GetCost(self, logits, cost, costKwargs):
        flattenLogits = tf.reshape(logits, [-1, self.classNumber])
        flattenLabels = tf.reshape(self.y, [-1, self.classNumber])
        
        if cost == 'cross-entropy':
            classWeights = costKwargs.pop('class-weights', None)
            
            if classWeights != None:
                self.globalWeights = classWeights
                
                logger.info(f'Class Weights = {classWeights}')
                
                weightMap = tf.multiply(flattenLabels, self.classWeights)
                weightMap = tf.reduce_sum(weightMap, axis=1)
                lossMap   = tf.nn.softmax_cross_entropy_with_logits(logits=flattenLogits, labels=flattenLabels)
                
                weightedLoss = tf.multiply(lossMap, weightMap)
                loss         = tf.reduce_mean(weightedLoss)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flattenLogits, labels=flattenLabels))
        elif cost == 'dice-coefficient':
            prediction   = PixelWiseSoftmaxActivation(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union        = 1E-5 + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss         = -(2 * intersection / union)
        else:
            raise ValueError(f'Unknown Cost Function: {cost}')
        
        regularizer = costKwargs.pop('regularizer', None)
        
        if regularizer != None:
            costSum  = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss    += regularizer * costSum
        
        return loss
    
    def _GetImageSummary(self, image, index=0):
        summary  = tf.slice(image, (0, 0, 0, index), (1, -1, -1, 1))
        summary -= tf.reduce_min(summary)
        summary /= tf.reduce_max(summary)
        summary *= 255
        
        width  = tf.shape(image)[1]
        height = tf.shape(image)[2]
        
        summary = tf.reshape(summary, tf.stack([width, height, 1]))
        summary = tf.transpose(summary, (2, 0, 1))
        summary = tf.reshape(summary, tf.stack([-1, width, height, 1]))
        
        return summary