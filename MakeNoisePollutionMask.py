import numpy as np
import matplotllib.pyplot as plt

stars = plt.imread('AverageImageCropped.png');
fgMask = plt.imread('MaskForAverageImageCropped.png');


# Iterate and find peaks
# Iterate over n x n grids to find maximum
# If overlap with mask dont do it there
gridSize = 200
overlap = 50
xGridIterMax = np.shape(stars)[0]/gridSize
yGridIterMax = np.shape(stars)[1]/gridSize

starMask = np.ones([np.shape(stars)[0],np.shape(stars)]) # Everything is set to white

#Iterate over all blocks
for xGridIter in range(0,xGridIterMax)
  for yGridIter in range(0,yGridIterMax)
    xBeg = (xGridIter)*gridSize
    xEnd = (xGridIter)*gridSize+gridSize
    yBeg = (yGridIter)*gridSize
    yEnd = (yGridIter)*gridSize+gridSize

    currChan1 = double(stars(xBeg:xEnd,yBeg:yEnd,0))
    currChan2 = double(stars(xBeg:xEnd,yBeg:yEnd,1))
    currChan3 = double(stars(xBeg:xEnd,yBeg:yEnd,2))

    blockMeanCh1 = np.mean(currChan1(:))
    blockMeanCh2 = np.mean(currChan2(:))
    blockMeanCh3 = np.mean(currChan3(:))
    blockStdevCh1 = np.std(currChan1(:))
    blockStdevCh2 = np.std(currChan2(:))
    blockStdevCh3 = np.std(currChan3(:))

    totalMaskCh1 = np.ones(np.shape(currChan1))*255
    totalMaskCh1[(currChan1-blockMeanCh1)>3.0*blockStdevCh1] = 0
    totalMaskCh2 = np.ones(np.shape(currChan2))
    totalMaskCh2[(currChan2-blockMeanCh2)>3.0*blockStdevCh2] = 0
    totalMaskCh3 = np.ones(np.shape(currChan3))
    totalMaskCh3[(currChan3-blockMeanCh3)>3.0*blockStdevCh3] = 0

    starMask[xBeg:xEnd,yBeg:yEnd] = np.min(np.min(totalMaskCh1,totalMaskCh2),totalMaskCh3);

plt.imshow(starMask)
input()
#imwrite(starMask,'mask4.png')
%% Perform a regression


xBeg = 1;
xEnd = size(stars,1);
yBeg = 1;
yEnd = size(stars,2);
currPatchChan1 = double(stars[xBeg:xEnd,yBeg:yEnd,0])
currPatchChan2 = double(stars[xBeg:xEnd,yBeg:yEnd,1])
currPatchChan3 = double(stars[xBeg:xEnd,yBeg:yEnd,2])
currPatchMask = double(starMask[xBeg:xEnd,yBeg:yEnd])
currPatchMask = min(currPatchMask,double(fgMask[:,:,1]))
X = np.zeros(np.shape(currPatchChan1)[0]*np.shape(currPatchChan1)[2])
iter = 1
for i=xBeg:xEnd
  for j=yBeg:yEnd
    X(iter,1) = i
    X(iter,2) = j
    X(iter,3) = 1
    iter = iter+1


y = [currPatchChan1(:),currPatchChan2(:),currPatchChan3(:)]
w = currPatchMask(:)
beta = lscov(X,y,w)

currPatchLightPolMask_col = X*beta;
temp1 = np.reshape(currPatchLightPolMask_col(:,1),np.shape(currPatchChan1)[0],np.shape(currPatchChan1)[1])
temp2 = np.reshape(currPatchLightPolMask_col(:,2),np.shape(currPatchChan1)[0],np.shape(currPatchChan1)[1])
temp3 = np.reshape(currPatchLightPolMask_col(:,3),np.shape(currPatchChan1)[0],np.shape(currPatchChan1)[1])
lightPolMask = zeros(size(stars));
lightPolMask(:,:,1) = temp1;
lightPolMask(:,:,2) = temp2;
lightPolMask(:,:,3) = temp3;

imshow(uint8(lightPolMask))
imwrite(stars-uint8(lightPolMask.*starMask),'Stars_WO_LightPolBackground.png')
imwrite(uint8(lightPolMask),'LightPolBackground.png')
