import tensorflow as tf

def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def ConvolutionLayer2D(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

def CropAndConcatTensor(x1, x2):
    x1Shape = tf.shape(x1)
    x2Shape = tf.shape(x2)
    
    offsets = [0, (x1Shape[1] - x2Shape[1]) // 2, (x1Shape[2] - x2Shape[2]) // 2, 0]
    size    = [-1, x2Shape[1], x2Shape[2], -1]
    
    return tf.concat([tf.slice(x1, offsets, size), x2], 3)

def CrossEntropyLoss(y, outputMap):
    return -tf.reduce_mean(y * tf.log(tf.clip_by_value(outputMap, 1E-10, 1.0)), name='cross_entropy')

def DeconvolutionLayer2D(x, weight, stride):
    inputShape  = tf.shape(x)
    outputShape = tf.stack([inputShape[0], inputShape[1] * 2, inputShape[2] * 2, inputShape[3] // 2])
    
    return tf.nn.conv2d_transpose(x, weight, outputShape, strides=[1, stride, stride, 1], padding='VALID')

def MaxPoolingLayer(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID')

def PixelWiseSoftmaxActivation(outputMap):
    exponentialMap          = tf.exp(outputMap)
    exponentialMapSum       = tf.reduce_sum(exponentialMap, 3, keep_dims=True)
    exponentialMapSumTensor = tf.tile(exponentialMapSum, tf.stack([1, 1, 1, tf.shape(outputMap)[3]]))
    
    return tf.div(exponentialMap, exponentialMapSumTensor)

def WeightVariable(shape, name=''):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())