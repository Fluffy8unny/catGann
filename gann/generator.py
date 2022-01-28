import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from gann.optimizers import crossEntropy

from image.meanColors import calcMeanImage
from settings import settings

def addBlock(model,filterSize,depth,upsampling=1):
    model.add(layers.Conv2DTranspose(depth, filterSize, strides=(upsampling, upsampling), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

def makeGeneratorModel():
    flattendSize          = settings.ann["generator"]["colorDescriptorSize"]**2
    noiseSize             = settings.ann["generator"]["noiseSize"]
    imgSize               = settings.img["imageSize"]
    channels              = settings.img["channels"]

    getDownsampledImgSize = lambda ds : [ s  //  (2**ds) for s in imgSize]
    sizeProduct           = lambda im : im[0]*im[1]
    
    model = tf.keras.Sequential()
    model.add(layers.Dense( sizeProduct(getDownsampledImgSize(2)) * 256, use_bias = False,
                            input_shape = ( flattendSize * channels + noiseSize,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((*getDownsampledImgSize(2), 256)))
    assert model.output_shape == (None, *getDownsampledImgSize(2), 256)  # Note: None is the batch size

    addBlock(model,(5,5),128)

    addBlock(model,(5,5),64,upsampling=2)

    model.add(layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, *getDownsampledImgSize(0), channels)

    return model

def generatorLoss(discOutput, genOutput, inputDistribution):
    discOutput      = crossEntropy(tf.ones_like(discOutput), discOutput)

    outDistribution = np.vstack(list( map( calcMeanImage,genOutput.numpy() ) ))
    colorEntropy    = crossEntropy(outDistribution, inputDistribution)

    return discOutput + colorEntropy