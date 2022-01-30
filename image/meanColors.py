import cv2
import numpy as np
import tensorflow as tf

from settings import settings

def calcMeanImageTF(img):
	descriptorSize   = settings.ann["generator"]["colorDescriptorSize"]
	return tf.image.resize(img, descriptorSize[:2])

def calcMeanImage(img):
	descriptorSize   = settings.ann["generator"]["colorDescriptorSize"]
	match img.dtype:
		case np.uint8:
			img = img.astype(np.float32) / 255.0

		case np.float64:
			img = img.astype(np.float32)

		case _:
			...

	return cv2.resize(img,descriptorSize[:2])

def calcColorDescriptor(img):
	return calcMeanImage(img).flatten()
	
