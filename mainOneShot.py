import sys
from PIL import Image
import numpy as np
from npbg import model
from matplotlib import cm
import matplotlib.pyplot as plt
import globalP as gp

frm = gp.Filename
f = open("../bbdd/" + frm + "/data", 'r')
params = list(f)  # Data init 0 , Data end 1, Image size 5,6.
params = map(int, params)

bgmodel = model(gp.imageSize, gp.N, 1)


fig = plt.figure()
graph1 = fig.add_subplot(111)

for i in range(params[0], params[1]):
	depthIm = np.asarray(Image.open("../bbdd/" + frm
                         + "/depthData/depth_" + str(i) + ".png"))
    
	colorIm = np.asarray(Image.open("../bbdd/" + frm + "/colorData/img_" + str(i) + ".Jpeg"))
	
	if depthIm is not None:
	
		if i == params[0]:
			
			print "Training frame: " + str(i)
			bgmodel.addFrame(depthIm)
			bgmodel.loadKernel()
		
		else:
			
			print "Image: "+str(i)
			resultat = bgmodel.subtractionP(depthIm)
			imgplot = plt.imshow(resultat, interpolation = 'none')
			imgplot.set_cmap('Spectral')
			plt.colorbar(imgplot, orientation='vertical')
			plt.savefig(str(i)+".png")
			plt.clf()
			
			if gp.evalOne  and i == params[0] + gp.N + 2:
				exit()
