import numpy as np
import matplotlib.pylab as plt

class KernelLUT:

	def __init__(self, s_max, s_min, bins, width):

		self.minsigma = s_min
		self.maxsigma = s_max
		self.bins = bins;
		self.width = width;
		
		self.kerneltable = np.zeros(bins * (2 * width + 1))
		self.kt = np.zeros((bins, width + 1))
		self.kernelsums= np.zeros(bins)

	def createTable(self):

		step = (self.maxsigma - self.minsigma) / self.bins
		
		sigma = self.minsigma

		for bins in range(0, self.bins):
			
			C1 = 1 / (np.sqrt( 2 * np.pi) * sigma)
			C2= -1 / (2 * sigma * sigma)
			
			b = (2 * self.width + 1) * bins;
			sums = 0;
			
			for x in range(0, self.width + 1):
				y = x / 1.0;
				v= C1 * np.exp(C2 * y * y)
				self.kerneltable[b + self.width + x] = v
				self.kerneltable[b + self.width - x] = v
				
				self.kt[bins, x] = v
				sums += 2 * v
				
			sums -= C1
			self.kernelsums[bins] = sums;

			#Normalization	
			for x in range(0, self.width + 1):
				v = self.kerneltable[b + self.width + x] / sums
				self.kerneltable[b + self.width + x] = v
				self.kerneltable[b + self.width - x] = v
				self.kt[bins, x] = v
			sigma += step


		#imgplot = plt.imshow(self.kt, interpolation = 'none')
		#imgplot.set_cmap('YlOrRd')
		#plt.colorbar(imgplot, orientation='vertical')
		#plt.savefig('kernel.png')	
