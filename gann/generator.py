from functools import reduce
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from gann.optimizers import crossEntropy

from image.meanColors import calcMeanImageTF
from settings import settings

def getBlock(filterSize,depth,upsampling=1):
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(depth, filterSize, strides=upsampling, padding='same', use_bias=False))
    result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result 

def makeGeneratorModel():
    descriptorSize         = settings.ann["generator"]["colorDescriptorSize"]
    inputChannelMultiplier = settings.ann["generator"]["inputChannelMultiplier"]

    channels              = descriptorSize[2]

    sizeProduct            = lambda im : reduce(lambda x,y: x*y, im)
    denseOutputWidth       = descriptorSize[2]*inputChannelMultiplier
    elementsInInput        = sizeProduct(descriptorSize)

    inputs = tf.keras.layers.Input(shape=descriptorSize)
   
    inputMLP = layers.Reshape([elementsInInput])(inputs)
    inputMLP = layers.Dense(elementsInInput*inputChannelMultiplier, use_bias=False, input_shape=descriptorSize)(inputMLP)
    inputMLP = layers.Reshape((*descriptorSize[:2],denseOutputWidth))(inputMLP)


    convLayerDepth  = settings.ann["generator"]["convLayerDepth"]
    scaling         = settings.ann["generator"]["upsamplingPerLayer"]    
    filterSize      = settings.ann["generator"]["filterSize"]

    numberOfLayers  = len(convLayerDepth)
    currentLayer = inputMLP

    cumulativeUpsampling = [1,1]
    for i,depth,s,fSize in zip(range(numberOfLayers), convLayerDepth,scaling,filterSize):
        convBlock    = getBlock(fSize,depth-denseOutputWidth,upsampling=s)
        currentLayer = convBlock(currentLayer)
        cumulativeUpsampling = [i*j for i,j in zip(cumulativeUpsampling,s)]
        if 0:
            passBlock    = getBlock((5,5),denseOutputWidth, upsampling=cumulativeUpsampling)(inputMLP)
            currentLayer = tf.keras.layers.Concatenate()([currentLayer, passBlock])

    outputs = layers.Conv2DTranspose(channels, (3, 3), padding='same', use_bias=False, activation='tanh')(currentLayer)
    return   tf.keras.Model(inputs=inputs, outputs=outputs)


def generatorLoss(discOutput, genOutput, inputDistribution):
    discOutput      = crossEntropy(tf.ones_like(discOutput), discOutput)

    outDistribution = calcMeanImageTF(genOutput)
    colorEntropy    = crossEntropy(outDistribution, inputDistribution)

    return discOutput + colorEntropy