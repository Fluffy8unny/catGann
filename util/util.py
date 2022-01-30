import os
from time import time
from pathlib import Path
from functools import reduce

from settings import settings

def checkFolders( folders ):
    for f in folders:
        Path(f).mkdir( parents=True,
                        exist_ok=True)

def addFolderToPath( path, folder ):
    return "".join([  path, os.path.sep, folder ])

class Timer:
    def __init__( self, message, addParam = {},timeSymbol ='t'):
        self.message    = message
        self.addParam   = addParam
        self.timeSymbol = timeSymbol

    def __enter__(self):
        self.start   = time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        argDict = {self.timeSymbol : time()-self.start} | self.addParam
        print(self.message.format(**argDict))
        return True

def getImageSize():
    descriptorSize  = settings.ann["generator"]["colorDescriptorSize"]
    scaling         = settings.ann["generator"]["upsamplingPerLayer"]
    imgSize         = [i for i in descriptorSize]    
    for s in scaling:
        imgSize[:2] = [i*j for i,j in zip(imgSize[:2],s)]

    return imgSize