import numpy as np
import cv2
import sys
import matplotlib.pylab as plt
from PIL import Image
import numpy.ma as ma
from scipy.stats import stats
from scipy import stats
import random
import matplotlib.mlab as mlab
from dTransform import dTransform
import statsmodels.api as sm
import csv
import os, os.path
import globalP as gp

class histogram:
	
	SIGMA_BINS = 50#(1500-650) +1
	SIGMA_MIN = 1.0
	SIGMA_MAX = 36.5
	DEFAULT_SIGMA = 5.0
	DEFAULT_BINS = 50 #(1500-650) +1
	
	def __init__(self, rows, cols, seqLenght, channels):
		
		self.rows = rows
		self.cols = cols
		self.channels = channels
		self.seqLenght = seqLenght

		self.kSize = (1500 - 650) + 1 # Depth bounds

		self.hist = np.zeros((self.kSize, rows, cols), dtype='int16')
		self.values = np.zeros((seqLenght, rows, cols),dtype='int16')
		
		self.mSize = (rows, cols)

		self.top        = np.zeros(self.mSize, dtype='uint8')
		self.medianBins = np.zeros(self.mSize, dtype = 'uint16') #([('r','uint8'),('g','uint8'),('b','uint8')]))
		self.accSum     = np.zeros(self.mSize, dtype = 'uint16')#([('r','uint8'),('g','uint8'),('b','uint8')]))
		self.stds       = np.zeros(self.mSize, dtype = 'float')
		
	
	def buildHist(self, model):

		k,j = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
	
		for i in range(1, self.seqLenght):

			x1 = model[i -1]
			x2 = model[i]

			idx = np.where(x2 == 650)

			x2[idx] = x1[idx]
			
			resultat = np.add(x1, -1 * x2)
			resultat = np.abs(resultat)

			self.hist[resultat, j, k] += 1
			
			#self._add2Hist(resultat)

	def depthHistogram(self):

		for i in range(1, self.kSize):
			value = i  + 650
			
			idx = np.where(self.medianBins == i)

			if len(idx[0]) > 0:

				brum =  np.zeros((self.kSize-1), dtype='float')
				total = 0

				for tpl in range(0, len(idx[0])):
					
					if self.hist[0, idx[0][tpl], idx[1][tpl]] < (self.seqLenght * 0.5):
					
						brum +=  self.hist[1:self.kSize, idx[0][tpl], idx[1][tpl]]
						total += self.seqLenght - self.hist[0, idx[0][tpl], idx[1][tpl]]
				
				brum = (brum * 1.0) / (total * 1.0)

				brum = np.nan_to_num(brum)

				self.freq[:,i] = brum
			
	def histograms(self, ntest):
		
		print "hISTOGRAMs"
		
		np.random.seed(seed=44)
		myfile = open('prove.csv', 'wb')
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		nprints = 0
		mostres = 200
		positions = np.zeros((2, mostres), dtype='uint16')
		vImg = np.zeros((self.rows, self.cols), dtype='uint16')
		examples = []
		#examples = [3,6,9,15,16,19,20,22,26,27,41]
		
		wr.writerow(examples)
		
		while nprints < mostres:

			nprints += 1
			#Random position
			i = np.random.randint(0,479)
			j = np.random.randint(0,639)

			s = set(self.values[:, i, j])
			nValues = len(s)

			if vImg[i,j] == 0 and nValues > 1:
		
				positions[:,nprints] = (i,j)
				
				vImg[i,j] = 1
				
				pValues = np.array(self.values[:, i, j])
				xno = pValues
				
				std = np.std(pValues)
				mu = np.mean(pValues)
				
				pValues = ( pValues - mu ) / float(std)
				
				#Write a file pixel Values
				if nprints in examples:
					wr.writerow([nprints])
					print str(nprints) +", p("+str(i)+","+str(j)+")"
					print self.values[:,i,j]
					print pValues
					
					wr.writerow(self.values[:,i,j])
					wr.writerow(pValues)

				#Normaltest
				z,pval = stats.normaltest(xnr)

				#Visualization stuff
				xmin = int(min(pValues))
				xmax = int(max(pValues))
				#Fig1
				fig = plt.figure(figsize=(14,8))
				ax = fig.add_subplot(121)
				xs = sorted(pValues)
				
				lspace = np.linspace(int(xmin-1),int(xmax+1),100)
				
				gkde=stats.gaussian_kde(xs)
				egkde = gkde.evaluate(lspace)

				if pval < 0.01 or np.isnan(pval): #ntest[i,j]):
					clr = "b"
				else:
					clr = "r"
				arsd = range(xmin, xmax)
				ax.plot(lspace, mlab.normpdf(lspace, np.mean(pValues), np.std(pValues)), color=clr, label = "Pvalue: "+str(pval)) #ntest[i,j])) 
				ax.hist(xnr, normed=True, histtype='bar', stacked=True, alpha=0.5, label="Standarized N(0,1)") # bins=(xmax-xmin)+1 
				ax.plot(lspace, egkde, label=' Standard kde', color="g")
				ax.legend()
				
                #fig2
				ax = fig.add_subplot(122)
				xmin = int(min(xno))
				xmax = int(max(xno))
								
				brum = self.hist[xmin:xmax+1,i,j]
				vrange = range(xmin,xmax+1)
				x = np.linspace(int(xmin-1),int(xmax+1),100)

				gkde=stats.gaussian_kde(sorted(xno),bw_method='silverman')

				print gkde
				egkde = gkde.evaluate(vrange)

				ax.bar(vrange, brum/sum(brum), align='center')
				ax.plot(vrange, egkde, label='kde', color="g")

				ax.legend()

				plt.savefig("dist"+str(nprints)+".png")
				plt.clf()
				plt.close()

	def _remFromHist(self, diffImage, mask):

		bins_1 = self.DEFAULT_BINS - 1
		k,j = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
		surpasing_index = diffImage > bins_1  # Where values surpass
		diffImage[surpasing_index] = bins_1
		
		ch_l = cv2.split(diffImage)
		for i in range(0, self.channels):
			self.hist[i,ch_l[i], j, k] -= 1
			
	def _add2Hist(self, image):
		
		k,j = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
		#self.hist[image, j, k] += 1
		self.hist[image, j, k] += 1

    #Find Hist Medians
    #TODO: computacionalment millorable
	def medianEstimation(self):
		#Lloc on trobarem la median		
		medianCount = (self.seqLenght - 1) / 2
		for j in xrange(self.cols):
			for i in xrange(self.rows):

				h = self.hist[:, i, j]
				#h[0] = 650 = nans


				bins, sums = self._binSearch(h[:], medianCount)

				self.medianBins[i, j] = bins #TODO: Prova lokendo
				self.accSum[i, j] = sums


	def _binSearch(self, hist, medianCount):

		suma = 0
		binc = 0

		while suma < medianCount:
			suma += hist[binc]
			binc += 1
		binc -= 1
		return binc, suma
	
	
	def stdEstimation(self):
		
		k,j = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
		k,j = np.meshgrid(np.arange(self.cols), np.arange(self.rows))

		binFactor = (self.DEFAULT_BINS - 1) / (self.SIGMA_MAX - self.SIGMA_MIN)
		medianCount = float(self.seqLenght - 1) / 2.0
		
		x1 = self.accSum - self.hist[self.medianBins, j, k]
		x2 = self.accSum
		#1.04 = 1 / (0.68*sqrt(2))
		aux = np.zeros((self.mSize),dtype='float')
		aux = ( self.medianBins - (x2.astype('float') - medianCount) / (x2.astype('float') - x1.astype('float')))
		v = 1.04 * aux

		v[v < self.SIGMA_MIN] = self.SIGMA_MIN
		#temp = np.where( v <= self.SIGMA_MIN)
		#v[temp] = self.SIGMA_MIN
		#temp = np.where(v >= self.SIGMA_MAX)
		#self.stds[temp] = self.DEFAULT_BINS -1
		#temp = np.where( v < self.SIGMA_MAX)

		self.stds = np.floor((v - self.SIGMA_MIN) * binFactor +0.5)

		self.stds[self.stds > self.SIGMA_MAX] = self.SIGMA_MAX

		#self.stds[self.stds == 0.0] = 33

		#temp = np.where(self.medianBins[:, :] == 0)
		
		#imgplot = plt.imshow(self.medianBins, interpolation = 'none')
		#imgplot.set_cmap('YlOrRd')
		#plt.colorbar(imgplot, orientation='vertical')
		#plt.show()




