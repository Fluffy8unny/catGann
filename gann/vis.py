import os

import cv2
import numpy as np
import pylab as py

from image.loader import loadImage
from image.meanColors import  getInputFromImage,calcMeanImage
from settings     import settings

def getDisplayableOutput(img):
	uint8img  = ( np.clip(img, 0.0, 1.0) * 255.0 ).astype(np.uint8)
	return cv2.cvtColor( uint8img, cv2.COLOR_BGR2RGB )

def getVisualizer():
    path           = settings.ds["testImage"]
    imageSize      = settings.img["imageSize"]
    ourputDir      = settings.train["output"]
   
    img       = loadImage(path,imageSize)
    inputData = getInputFromImage(calcMeanImage(img))

    def saveTestImage(generator,epoch):
        generatedImage = generator( inputData[np.newaxis,...], training = False )
        
        py.subplot(1,2,1)
        py.title("color input")
        py.imshow(getDisplayableOutput(img))
        
        py.subplot(1,2,2)
        py.imshow(getDisplayableOutput(generatedImage[0,...])) #network outputs [imgsize,w,h,c]
        py.title(f"generated")
        
        py.savefig(f"{ourputDir}{os.path.sep}vis{epoch}.jpg")
        py.close()
    return saveTestImage