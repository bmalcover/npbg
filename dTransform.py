import numpy as np


class dTransform:
	#Depth2dispersion
	def __init__(self):
		
		length = 10000
		self.xd = np.zeros(shape=(length,), dtype=float, order='C')
		self.xd[0] = 650
		
		for i in xrange(1, length):
			calc = round(1000 - (3.3309495161 * i)) / (-0.0030711016 * i)
			if calc > 650 and calc < 1500:
				self.xd[i] = calc
			else:
				self.xd[i] = 650

	def d2d(self,v):
		return self.xd[v]

	def depth2dispersion(self, depthIm):

		vfunc = np.vectorize(self.d2d, otypes= ['uint16'])

		return vfunc(depthIm)
