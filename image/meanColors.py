import cv2
import numpy as np

from settings import settings

def calcMeanImage(img):
	descriptorSize   = settings.ann["generator"]["colorDescriptorSize"]
	match img.dtype:
		case np.uint8:
			img = img.astype(np.float32) / 255.0

		case np.float64:
			img = img.astype(np.float32)

		case _:
			...

	return cv2.resize(img,[descriptorSize,descriptorSize]).flatten()

def getInputFromImage(meanImage):
	noiseSize      = settings.ann["generator"]["noiseSize"]

	inp            = np.hstack( [ meanImage
						 		 ,np.random.rand(noiseSize).astype(np.float32)
						 		  ])
	return inp