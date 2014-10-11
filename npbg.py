import numpy as np
import cv2
from chistogram import histogram
from kernel import KernelLUT
from dTransform import dTransform
import matplotlib.pylab as plt
import globalP as gp

class model:

	def __init__(self, imageSize, modelLength, channels):
		self.model = []
		self.lFrame = 0
		self.modelLength = modelLength
		self.imageSize = imageSize
		self.channels = channels
		self.previous = np.zeros((imageSize))
        
		self.hist = histogram( imageSize[0], imageSize[1], modelLength, channels)
		self.positions = np.zeros(self.hist.kSize, dtype=[('x','i8'),('y','i8')])
		
	def addFrame(self, image):

		dt = dTransform()
		tImage = dt.depth2dispersion(image)

		if len(self.model) < self.modelLength:
			self.model.append(tImage)
	
	def estimation(self, cIm):
		
		self.hist.buildHist(self.model)
		self.hist.medianEstimation()
		self.hist.stdEstimation()
		#self.hist.depthHistogram()
		#self.hist.genericKDE()
        #
        #self.knl = KernelLUT(self.hist.SIGMA_MAX, self.hist.SIGMA_MIN, self.hist.SIGMA_BINS, 25000)
        #self.knl.createTable()


	def equalityTest(self):

		print "Eq Test"
		x = range(650,1501)
		
		for i in range(1, self.hist.kSize):
			value = i  + 650
		#value = i  + 650
			idx = np.where(self.hist.medianBins == i)
		#idx = np.where(self.model[0] == value)

			print "Value: "+str(value)+" - "+str(len(idx[0]))
			if len(idx[0]) > 0:

				fig = plt.figure(figsize=(14,8))
				ax = fig.add_subplot(111)
				brum =  np.zeros((self.hist.kSize-1), dtype='int32')
				total = 0

				for tpl in range(0, len(idx[0])):
					
					if self.hist.hist[0, idx[0][tpl], idx[1][tpl]] < (self.hist.seqLenght * 0.5):
					#ax.plot(x, self.hist.kernel[:, idx[0][tpl], idx[1][tpl]] / self.hist.hist[:, idx[0][tpl], idx[1][tpl]])
					#xno = np.array(self.hist.values[:, idx[0][tpl], idx[1][tpl]])
					#ax.set_xlim(value-40, value+40)
				
					#ax = fig.add_subplot(122)
					#xmin = int(min(xno))
					#xmax = int(max(xno))
						brum +=  self.hist.hist[1:self.hist.kSize, idx[0][tpl], idx[1][tpl]]
						
						total += self.hist.seqLenght - self.hist.hist[0, idx[0][tpl], idx[1][tpl]]
				
				arsd = range(651, self.hist.kSize+650)
					#arsd = range(xmin,xmax)
				brum = (brum * 1.0) / (total * 1.0)
				ax.bar(arsd, brum, align='center', label=str(total))
				ax.legend()
				ax.set_xlim(i+650-10, i+ 650 + 10)

				#ax.cla()
				plt.savefig("eval"+str(650+i)+".png")
				plt.clf()
				plt.close()

		

	def loadKernel(self):

		print "Loading Kernel"
		self.knl =np.load('knl.npy')


	def saveKernel(self):

		knl = np.zeros((self.hist.kSize, self.hist.kSize + 1), dtype= float)
		
		for i in range(0,self.hist.kSize):
			ii = self.positions[i]['x']
			jj = self.positions[i]['y']

			if ii != 0 and jj != 0:
			
				knl[i,1:self.hist.kSize + 1] = self.hist.kernel[:,ii,jj]
				knl[i,0] = 1
		print "Saving Kernel file"
		np.save('knl', knl)
		
		if gp.debug:
			rng = range(650,1500+1)
			plt.plot(rng,knl[:,0])
			plt.show()

	def subtractionP(self,cImage):

		dt = dTransform()
		i2 = dt.depth2dispersion(cImage)
		
		i2 = i2.astype(int)
		
		resultat = np.zeros((640, 480))
		resultat = self.hist.kernel[i2-650, self.hist.medianBins]

		#amax = np.amax(resultat[resultat != 1.0])
		#resultat[resultat == 1.0] = amax

		#resultat[ i2 == 650] = amax

		#resultat[self.hist.kernel[self.hist.medianBins-650] == 0] = -2
		#nans = np.where(i2 == 650)


		return resultat


	def subtraction(self, cImage):
		
		k,j = np.meshgrid(np.arange(480), np.arange(640))
		i3 = np.zeros((480, 640),dtype='int')
		dt = dTransform()
		i2 = dt.depth2dispersion(cImage)
		i2 = i2.astype(int)
        #th = 0
        #pSum = np.ndarray((480, 640))
		resultat = np.zeros((640, 480))
		resultat.fill(255)
		i3 = np.copy(i2)
        #aux = np.where(self.hist.kernel[np.transpose(i3)-650, k, j] > 0.0005)

        #for i in range(0, self.modelLength):
        #    
        #    k1 = np.zeros((480,640))
        #    
        #    model = self.model[i]
        #    subs =  abs(model[k, j] - cImage[k, j])
        #    std = self.hist.stds[0, k, j]

        #    k1[k, j] = np.squeeze(self.knl.kt[std.astype(np.uint8), subs])
        #    
        #    
        #    k1 = np.transpose(k1)
        #    pSum[k,j] += k1[j, k] #* k2[j, k]* k3[j, k]
        #    
        #    #k, j = np.where(pSum < th)
            
        #aux = np.where((pSum / self.modelLength) == th)
		resultat = self.hist.kernel[np.transpose(i3)-650, k, j]
		#reuse = 0
		#for i in range(0,480):
		#	for j in range(0,640):
		#		value = i2[i,j] - 650
		#		if self.positions['x'][value] == 0 and self.positions['y'][value] == 0:
		#			#resultat[j,i] = self.hist.kernel[value, i, j]
		#			if self.hist.kernel[value, i, j] > 0.1:
		#				resultat[j,i] = 0
		#				
		#			self.positions['x'][value] = i
		#			self.positions['y'][value] = j
		#		else:# self.positions['x'][value] != 0 and self.positions['y'][value] == 0:
		#			reuse = reuse + 1
		#			ii = self.positions[value]['x']
		#			jj = self.positions[value]['y']	
		#			#resultat[j,i] = self.hist.kernel[value, i, j]
		#			if self.hist.kernel[value, ii, jj] > 0.10:
		#				resultat[j,i] = 0
		
		nans = np.where(i2 == 650)
		
		resultat = np.transpose(resultat)
		resultat[nans] = -1.0 #self.previous[nans]
		self.previous = resultat
        # Show kernel distribution in random samples 
		if gp.debug:
			np.random.seed(seed=44)
			rnd = 0
			
			while rnd < 60:
				i = np.random.randint(0,479)
				j = np.random.randint(0,639)
				
				if resultat[j,i] != gp.debugAll:
					
					rnd = rnd + 1

					fig = plt.figure(figsize=(14,8))
					ax = fig.add_subplot(121)

					x = range(650,1501)
					
					ax.plot(x, self.hist.kernel[:, i, j], label='Standard kde', color="g")
					ax.plot(i2[i, j], self.hist.kernel[i2[i, j]-650, i, j], marker='o', c='b', label=str(i2[i, j]))
					
					ax.set_xlim(min(self.hist.values[:, i, j]) - 10 , max(self.hist.values[:, i, j]) + 10)
					
					ax.legend()
					
					ax = fig.add_subplot(122)
					
					xno = np.array(self.hist.values[:, i, j])
					xmin = int(min(xno))
					xmax = int(max(xno))
					brum = self.hist.hist[:, i, j]
					arsd = range(0, self.hist.kSize)
					#arsd = range(xmin, xmax+1)

					ax.bar(arsd, brum/sum(brum), align='center')
					plt.savefig("eval"+str(rnd)+".png")
					plt.clf()
					plt.close

        #x = aux[0][0]
        #y = aux[1][0]
        #print "("+str(x)+","+str(y)+")"
        #print "Hist: "+str(self.hist.hist[0,:,x,y])
		#print "td bin : " +str(self.hist.stds[0,x,y])
        #print "bin: "+str(self.hist.medianBins[0,x,y])
        #print "Image val: "+str(cImage[x,y])

        #s = [self.model[i][x,y] for i in range(0,self.modelLength)]
        #print "Subs values: "+str(np.abs(s-cImage[x,y]))
        #print "Model values: "+str(s)

		return resultat #pSum / self.modelLength


