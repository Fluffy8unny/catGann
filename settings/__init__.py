import yaml
from os.path import sep

def load(path):
	with open(path,'r') as f: 
		return yaml.load(f, Loader=yaml.FullLoader)

class Settings:
	def __init__(self,**kwargs):
		for k,p in kwargs.items():
			setattr(self,k,load(p))

basePath     = r"settings"
makeAbsolute = lambda x : f"{basePath}{sep}{x}"

settings = Settings(  ann   = makeAbsolute("annSettings.yml")
					, img   = makeAbsolute("imageSettings.yml") 
					, ds    = makeAbsolute("dataset.yml")
					, train = makeAbsolute("train.yml")
					)