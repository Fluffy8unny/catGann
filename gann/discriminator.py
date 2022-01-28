import tensorflow as tf
from tensorflow.keras import layers

from gann.optimizers import crossEntropy

from settings import settings

def makeDiscriminatorModel():
    imgSize               = settings.img["imageSize"]
    channels              = settings.img["channels"]

    model = tf.keras.Sequential()
    model.add( layers.Conv2D( 64, (5, 5), strides = (2, 2), padding='same',
                                          input_shape = [imgSize[0], imgSize[1], channels]) )
    model.add( layers.LeakyReLU() )
    model.add( layers.Dropout(0.3))

    model.add( layers.Conv2D( 128, (5, 5), strides=(2, 2), padding='same') )
    model.add( layers.LeakyReLU() )
    model.add( layers.Dropout(0.3) )

    model.add( layers.Flatten() )
    model.add( layers.Dense(1) )

    return model

def discriminatorLoss(real, fake):
    realLoss = crossEntropy( tf.ones_like(real), real)
    fakeLoss = crossEntropy( tf.zeros_like(fake), fake)
    return realLoss + fakeLoss