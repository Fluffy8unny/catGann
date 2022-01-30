import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import get_graph
from tensorflow.python.ops import summary_ops_v2

from image.loader import loadImage
from image.meanColors import  calcMeanImage
from settings     import settings
from util.util    import getImageSize

@tf.function
def traceme(model,img):
    return model(img)
    
def plotModelGraph( name, model,testInput, logDir ):
    writer      = tf.summary.create_file_writer(logDir)
    tf.summary.trace_on(graph=True, profiler=True)
    traceme(model,testInput)
    with writer.as_default():
        tf.summary.trace_export(name, step=0,  profiler_outdir=logDir)

def getRGB(img):
    normedImg = ( img + 1.0 ) / 2.0
    rgb = np.stack([normedImg[...,i] for i in [2,1,0]],axis=-1)
    return rgb

def getImageWriter(logDir):
    path           = settings.ds["testImage"]
    imageSize      = getImageSize()
    ourputDir      = settings.train["output"]
   
    img         = loadImage(path,imageSize[:2])
    meanImg     = calcMeanImage(img)

    writer      = tf.summary.create_file_writer(logDir)
    
    with writer.as_default():
        tf.summary.image("inputImage",      getRGB(img)[np.newaxis,...],     step = 1)
        tf.summary.image("colorDescriptor", getRGB(meanImg)[np.newaxis,...], step = 1)
 
    def saveTestImage(generator,epoch):
        generatedImage = generator( meanImg[np.newaxis,...], training = False )
        with writer.as_default():
            tf.summary.image( "generatedOutput" , getRGB(generatedImage), step = epoch )

    return saveTestImage