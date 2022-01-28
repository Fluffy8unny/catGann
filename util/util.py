from time import time

class Timer:
	def __init__(self,message,addParam):
		self.message  = message
		self.addParam = addParam
	
	def __enter__(self):
		self.start   = time()
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		argDict = {'t' : time()-self.start} | self.addParam
		print(self.message.format(**argDict))
		return True
