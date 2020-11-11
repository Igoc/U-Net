from NeuralNetwork.Logger  import logger
from NeuralNetwork.Utility import CropData

import numpy               as np
import os
import shutil
import sys
import tensorflow          as tf

class Trainer(object):
    def __init__(self, network, optimizer='adam', crop=False, **kwargs):
        self.network                          = network
        self.optimizer, self.learningRateNode = self._GetOptimizer(optimizer, tf.Variable(0), **kwargs)
        self.normGradientsNode                = tf.Variable(tf.constant(0.0, shape=[len(self.network.gradientsNode)]))
        self.crop                             = crop
    
    def Train(self, dataProvider, trainingIteration, outputPath, logPath, maxEpoch=100, dropout=0.5, displaySteps=1, restoration=False, graph=False):
        self._Initialize(outputPath, logPath, restoration)
        
        averageGradients = None
        minAverageLoss   = sys.float_info.max
        
        with tf.Session() as session:
            if graph == True:
                tf.train.write_graph(session.graph_def, outputPath, 'graph.pb', False)
            
            session.run(tf.global_variables_initializer())
            
            if restoration == True:
                checkpoint = tf.train.get_checkpoint_state(outputPath)
                
                if checkpoint != None and checkpoint.model_checkpoint_path != None:
                    self.network.LoadModel(session, checkpoint.model_checkpoint_path)
            
            x, y = dataProvider()
            
            prediction = session.run(self.network.prediction, feed_dict={
                self.network.x: x,
                self.network.y: y,
                self.network.keepProbability: 1.0,
                self.network.classWeights: self.network.globalWeights
            })
            
            predictionShape = prediction.shape
            
            summaryWriter = tf.summary.FileWriter(logPath, session.graph)
            
            logger.info('Start Optimization')
            
            for epoch in range(maxEpoch):
                totalLoss = 0
                
                for step in range(epoch * trainingIteration, (epoch + 1) * trainingIteration):
                    x, y = dataProvider()
                    
                    if self.crop == True:
                        y = CropData(y, predictionShape)
                    
                    _, loss, learningRate, gradients = session.run([self.optimizer, self.network.cost, self.learningRateNode, self.network.gradientsNode], feed_dict={
                        self.network.x: x,
                        self.network.y: y,
                        self.network.keepProbability: dropout,
                        self.network.classWeights: self.network.globalWeights
                    })
                    
                    if averageGradients == None:
                        averageGradients = [np.zeros_like(gradient) for gradient in gradients]
                    
                    for index in range(len(gradients)):
                        averageGradients[index] = averageGradients[index] * (1.0 - 1.0 / (step + 1)) + gradients[index] / (step + 1)
                    
                    normGradients = [np.linalg.norm(gradient) for gradient in averageGradients]
                    self.normGradientsNode.assign(normGradients).eval()
                    
                    if step % displaySteps == 0:
                        self._ShowMinibatchStats(session, summaryWriter, step, x, y)
                    
                    totalLoss += loss
                
                averageLoss = totalLoss / trainingIteration
                
                self._ShowEpochStats(epoch, averageLoss, learningRate)
                
                if epoch == 0 or averageLoss < minAverageLoss:
                    self.network.SaveModel(session, f'{outputPath}/model.cpkt')
                    minAverageLoss = averageLoss
            
            logger.info('Optimization Finished')
    
    def _Initialize(self, outputPath, logPath, restoration):
        tf.summary.scalar('Learning-Rate', self.learningRateNode)
        tf.summary.scalar('Accuracy', self.network.accuracy)
        tf.summary.scalar('Loss', self.network.cost)
        tf.summary.scalar('Cross-Entropy', self.network.crossEntropy)
        
        self.summaries = tf.summary.merge_all()
        
        if restoration == False:
            shutil.rmtree(outputPath, ignore_errors=True)
        
        if os.path.exists(outputPath) == False:
            os.makedirs(outputPath)
        
        if os.path.exists(logPath) == False:
            os.makedirs(logPath)
    
    def _ShowEpochStats(self, epoch, averageLoss, learningRate):
        logger.info(f'Epoch {epoch}, Average Loss = {averageLoss:.4f}, Learning Rate = {learningRate:.7f}')
    
    def _ShowMinibatchStats(self, session, summaryWriter, step, x, y):
        summary, loss, accuracy, prediction = session.run([self.summaries, self.network.cost, self.network.accuracy, self.network.prediction], feed_dict={
            self.network.x: x,
            self.network.y: y,
            self.network.keepProbability: 1.0,
            self.network.classWeights: self.network.globalWeights
        })
        
        summaryWriter.add_summary(summary, step)
        summaryWriter.flush()
        
        minibatchError = self._GetErrorRate(prediction, y)
        
        logger.info(f'Iteration {step}, Minibatch Loss = {loss:.4f}, Training Accuracy = {accuracy:.4f}, Minibatch Error = {minibatchError:.1f}%')
    
    def _GetOptimizer(self, optimizer, globalStep, **kwargs):
        learningRate = kwargs.pop('learning-rate', 0.0001)
        decaySteps   = kwargs.pop('decay-steps', 10)
        decayRate    = kwargs.pop('decay-rate', 0.99)
        
        learningRateNode = tf.train.exponential_decay(learningRate, globalStep, decaySteps, decayRate, True)
        
        if optimizer == 'momentum':
            momentum  = kwargs.pop('momentum', 0.2)
            optimizer = tf.train.MomentumOptimizer(learningRateNode, momentum, **kwargs).minimize(self.network.cost, globalStep)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learningRateNode, **kwargs).minimize(self.network.cost, globalStep)
        else:
            raise ValueError(f'Unknown Optimizer: {optimizer}')
        
        return optimizer, learningRateNode
    
    def _GetErrorRate(self, prediction, label):
        return 100.0 - 100.0 * np.sum(np.argmax(prediction, 3) == np.argmax(label, 3)) / (prediction.shape[0] * prediction.shape[1] * prediction.shape[2])