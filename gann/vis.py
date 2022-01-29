import os

import cv2
import numpy as np
import tensorflow as tf

from image.loader import loadImage
from image.meanColors import  getInputFromImage,calcMeanImage
from settings     import settings

def getRGB(img):
    rgb = np.stack([img[...,i] for i in [2,1,0]],axis=-1)
    return rgb

def getImageWriter(logDir):
    path           = settings.ds["testImage"]
    imageSize      = settings.img["imageSize"]
    ourputDir      = settings.train["output"]
   
    img         = loadImage(path,imageSize[:2])
    meanImg     = calcMeanImage(img)
    inputData   = getInputFromImage(meanImg.flatten())

    writer      = tf.summary.create_file_writer(logDir)
    
    with writer.as_default():
        tf.summary.image("inputImage",      getRGB(img)[np.newaxis,...],     step = 1)
        tf.summary.image("colorDescriptor", getRGB(meanImg)[np.newaxis,...], step = 1)
 
    def saveTestImage(generator,epoch):
        generatedImage = generator( inputData[np.newaxis,...], training = False )
        with writer.as_default():
            tf.summary.image( "generatedOutput" , getRGB(generatedImage), step = epoch )

    return saveTestImage