import os

import cv2
import numpy as np
import tensorflow as tf

from image.loader import loadImage
from image.meanColors import  getInputFromImage,calcMeanImage
from settings     import settings

def getDisplayableOutput(img):
	uint8img  = ( np.clip(img, 0.0, 1.0) * 255.0 ).astype(np.uint8)
	return cv2.cvtColor( uint8img, cv2.COLOR_BGR2RGB )

def getImageWriter(logDir):
    path           = settings.ds["testImage"]
    imageSize      = settings.img["imageSize"]
    ourputDir      = settings.train["output"]
   
    img         = loadImage(path,imageSize[:2])
    meanImg     = calcMeanImage(img)
    inputData   = getInputFromImage(meanImg.flatten())
    
    writer      = tf.summary.create_file_writer(logDir)
    
    def saveTestImage(generator,epoch):
        generatedImage = generator( inputData[np.newaxis,...], training = False )
        with writer.as_default():
            tf.summary.image("generatedOutput", generatedImage, step=epoch)
    return saveTestImage