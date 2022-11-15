import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

stars = plt.imread('FromCamera102D3400/_DSC0365.jpg')
fgMask = plt.imread('MaskForAverageImageCropped.png')

# Iterate and find peaks
# Iterate over n x n grids to find maximum
# If overlap with mask dont do it there
gridSize = 200
overlap = 50
xGridIterMax = np.shape(stars)[0]//gridSize
yGridIterMax = np.shape(stars)[1]//gridSize

starMask = np.ones([np.shape(stars)[0],np.shape(stars)[1]]) # Everything is set to white

#Iterate over all blocks
for xGridIter in range(0,xGridIterMax):
    for yGridIter in range(0,yGridIterMax):
        xBeg = (xGridIter)*gridSize
        xEnd = (xGridIter)*gridSize+gridSize
        yBeg = (yGridIter)*gridSize
        yEnd = (yGridIter)*gridSize+gridSize

        currChan1 = stars[xBeg:xEnd,yBeg:yEnd,0].astype(float)
        currChan2 = stars[xBeg:xEnd,yBeg:yEnd,1].astype(float)
        currChan3 = stars[xBeg:xEnd,yBeg:yEnd,2].astype(float)

        blockMeanCh1 = np.mean(currChan1[:])
        blockMeanCh2 = np.mean(currChan2[:])
        blockMeanCh3 = np.mean(currChan3[:])
        blockStdevCh1 = np.std(currChan1[:])
        blockStdevCh2 = np.std(currChan2[:])
        blockStdevCh3 = np.std(currChan3[:])

        totalMaskCh1 = np.ones(np.shape(currChan1))*255
        totalMaskCh1[(currChan1-blockMeanCh1)>3.0*blockStdevCh1] = 0
        totalMaskCh2 = np.ones(np.shape(currChan2))
        totalMaskCh2[(currChan2-blockMeanCh2)>3.0*blockStdevCh2] = 0
        totalMaskCh3 = np.ones(np.shape(currChan3))
        totalMaskCh3[(currChan3-blockMeanCh3)>3.0*blockStdevCh3] = 0
        starMask[xBeg:xEnd,yBeg:yEnd] = np.minimum(np.minimum(totalMaskCh1,totalMaskCh2),totalMaskCh3)

plt.imsave('dum.png',starMask)
#imwrite(starMask,'mask4.png')
#Perform a regression


xBeg = 0
xEnd = stars.shape[0]
yBeg = 0
yEnd = stars.shape[1]
currPatchChan1 = stars[xBeg:xEnd,yBeg:yEnd,0].astype(float)
currPatchChan2 = stars[xBeg:xEnd,yBeg:yEnd,1].astype(float)
currPatchChan3 = stars[xBeg:xEnd,yBeg:yEnd,2].astype(float)
currPatchMask = starMask[xBeg:xEnd,yBeg:yEnd].astype(float)


currPatchMask = np.minimum(currPatchMask,fgMask[:,:,0].astype(float))
X = np.zeros((np.shape(currPatchChan1)[0]*np.shape(currPatchChan1)[1],3))
iter = 0
for i in range(xBeg,xEnd):
    for j in range(yBeg,yEnd):
        X[iter,0] = i
        X[iter,1] = j
        X[iter,2] = 1
        iter = iter+1


Y = np.transpose(np.asarray([currPatchChan1.flatten('F'),currPatchChan2.flatten('F'),currPatchChan3.flatten('F')]))
W = currPatchMask.flatten('F')

reg = LinearRegression().fit(X, Y, W)

# W = np.sqrt(np.diag(W))
# Aw = np.dot(W,A)
# Bw = np.dot(B,W)
# X = np.linalg.lstsq(Aw, Bw)


# WLS.fit(A, B, sample_weight=W)

#beta = lscov(X,y,w)

currPatchLightPolMask_col = reg.predict(X)
temp1 = currPatchLightPolMask_col[:,0].reshape((np.shape(currPatchChan1)[0],np.shape(currPatchChan1)[1]),order='F')
temp2 = currPatchLightPolMask_col[:,1].reshape((np.shape(currPatchChan1)[0],np.shape(currPatchChan1)[1]),order='F')
temp3 = currPatchLightPolMask_col[:,2].reshape((np.shape(currPatchChan1)[0],np.shape(currPatchChan1)[1]),order='F')
lightPolMask = np.zeros(np.shape(stars))

#print(np.shape(temp1),np.shape(stars))
lightPolMask[:,:,0] = np.multiply(temp1,starMask)
lightPolMask[:,:,1] = np.multiply(temp2,starMask)
lightPolMask[:,:,2] = np.multiply(temp3,starMask)

plt.imsave('dum2.png',np.clip(stars-lightPolMask,0,1))

# imshow(uint8(lightPolMask))
# imwrite(stars-uint8(lightPolMask*starMask),'Stars_WO_LightPolBackground.png')
# imwrite(uint8(lightPolMask),'LightPolBackground.png')
