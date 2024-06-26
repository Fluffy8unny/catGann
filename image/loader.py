import os
import re
from os.path import sep
from math import ceil

import cv2 
import numpy as np
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor,wait
from image.meanColors import calcMeanImage 
from settings import settings
from util.util import getImageSize

def getSlices(off,out):
    return [slice(i,i+j) for i,j in zip(off,out)]

def resize(image,outputShape):
    quots      = [ j / i for i,j in zip(image.shape,outputShape)]
    outSize    = [ ceil( i * max(*quots) ) for i,j in zip(image.shape,outputShape)]
    offset     = [ (i-j) // 2 for i,j in zip(outSize,outputShape)]

    slices     = getSlices(offset,outputShape)
    return cv2.resize(image,outSize[::-1])[slices[0],slices[1],...]

def loadImage(path,imageSize):
    loadedImg = cv2.imread(path)
    resized   = resize(loadedImg,imageSize)

    return resized.astype(np.float32) / 127.5 - 1.0

def checkSuffix(path):
    match (res := re.match( r".*(\.[a-z]+)$", path)):
        case None:
            return False
        case _:
            return res.groups(1)[0] in settings.ds["suffix"]

def imageGenerator( batchSize, imageSize, path ):
    files      = [ p for p in os.listdir(path) if  checkSuffix(p) ]
    paths      = np.array( list( map( lambda f: "".join([path,sep,f]), files ) ) )
    np.random.shuffle(paths)

    getFutures = lambda i: [ executor.submit( loadImage, c, imageSize )
                                         for c in paths[ i : i + batchSize ] ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures  = getFutures(0)
        for i in np.arange(batchSize,paths.shape[0],batchSize,dtype=np.int):
            wait(futures)

            images    = [ f.result()               for f   in futures  ]
            meanData  = [ calcMeanImage(img) for img in images   ]

            futures   = getFutures(i)
            yield images, meanData

def getDataset( batchSize, path):
    imageSize          = getImageSize()
    imgSize, channels  = imageSize[:2],imageSize[2]
    descSize           = settings.ann["generator"]["colorDescriptorSize"]

    return tf.data.Dataset.from_generator(
                                            lambda : imageGenerator( batchSize, imgSize, path ),
                                            output_types  = ( tf.float32, tf.float32 ),
                                            output_shapes = ( [batchSize, *imgSize, channels],
                                                              [batchSize, *descSize] )
                                         )