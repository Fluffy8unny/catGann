from gann.train   import train
from image.loader import getDataset
from settings     import settings

train( getDataset(  settings.train["batchSize"]
			      , settings.ds["path"] 
       ))