import numpy as np
import cv2
import random
import math
import time

noise_small=0.001
noise_medium=0.027
noise_high=0.07
noise_ultrahigh=0.5


def add_g_noise(image,var=1000):
    mean = 0
    sigma = var ** 0.5
    gauss=np.zeros(image.shape,np.uint8)
    cv2.randn(gauss,mean,sigma)
    noisy=np.add(image, gauss)
    return noisy


def add_sp_noise(image,prob=0.027):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def MSE(image,noisy):
    return np.square(np.subtract(image, noisy)).mean()

def PSNR(image, noisy):
    mse = MSE(image,noisy)
    if mse == 0:
     return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def showImg(image,title):
    screen_res = 512, 512
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, window_width, window_height)
    cv2.imshow(title, image)


def AdaptiveLocalFilter(image,kernelSize=3):
    lVars=np.zeros(image.shape)
    lMeans=np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel=[]
            for k in range(kernelSize):
                for t in range(kernelSize):
                    x = i + (k - 1)
                    y = j + (t - 1)
                    if ((x < image.shape[0] and x >= 0) and (y < image.shape[1] and y >= 0)):
                        kernel.append(image[x][y])
            lVars[i][j]=np.var(kernel)
            lMeans[i][j]=np.mean(kernel)

    nVar= lVars.sum()/(image.shape[0]*image.shape[1])

    for x in range(image.shape[0]):
        for z in range(image.shape[1]):
            lVars[x][z]=max(lVars[x][z],nVar)

    filtered = nVar / lVars
    filtered = filtered*(image - lMeans)
    filtered = image - filtered.astype(np.uint8)


    return filtered



def AdaptiveMedianFilter(img,filtered,sMax=7):
    width=img.shape[0]
    height=img.shape[1]
    filterSize=3
    for i in range(width):
        for j in range(height):
            kernel =[]
            filterSize=3
            zxy = img[i][j]
            result = zxy
            while(filterSize<=sMax):
                for k in range(filterSize):
                    for t in range(filterSize):
                        x = i + (k - 1)
                        y = j + (t - 1)
                        if ((x < img.shape[0] and x >= 0) and (y < img.shape[1] and y >= 0)):
                            kernel.append(img[x][y])

                kernel.sort()
                length=len(kernel)
                med  = length//2
                zmin = kernel[0]
                zmax = kernel[length-1]
                zmed = kernel[med]
                if(zmed<zmax and zmed > zmin or (filterSize+2 > sMax) ):
                    if(zxy>zmin and zxy<zmax):
                        result = zxy
                    else:
                        result = zmed
                    break
                else:
                    filterSize += 2
            filtered[i][j]=result

def NoiseProbIndex(index):
    if(index == 4):
        return noise_small
    if (index == 5):
        return noise_medium
    else:
        return noise_high








"""

# ------------------------- TESTING ------------------------------
kernel = (3, 3)
variance = 500.0
noiseT = 0
runTime=0
psnr=0
mse=0
for k in range(12):
    image = cv2.imread('./dataset/' + str(k + 1) + '.jpg', cv2.IMREAD_GRAYSCALE)
    filtered = np.zeros(image.shape, np.uint8)
    if (k > 3 and k < 8):
        kernel = (5, 5)
        noiseT = 1
    if (k > 7):
        kernel = (7, 7)
        noiseT = 2
    if (noiseT == 0):
        noisy = add_g_noise(image, variance * (k + 1))
    if (noiseT == 1):
        noisy = add_sp_noise(image, NoiseProbIndex(k))
    if (noiseT == 2):
        noisy = add_g_noise(image, variance * (k + 1 - 8))
        noisy = add_sp_noise(noisy, NoiseProbIndex(k - 4))

    startTime = time.time()
    #filtered = cv2.boxFilter(noisy, -1, kernel, None, (-1, -1), 1, cv2.BORDER_DEFAULT)
    #filtered = cv2.GaussianBlur(noisy, kernel, kernel[0], None, kernel[0])  # FILTER (image,kernel size,sigmax,-,sigmay)
    #AdaptiveMedianFilter(noisy, filtered, 7)  # ------ (image,dest,maxK=7)
    filtered = cv2.medianBlur(noisy, kernel[0])  # FILTER  (image,kernel size)
    
    #diagonal = (2 * ((image.shape[0] * image.shape[1]) ** 2)) **.5
    #diagonal= (diagonal*((k+1)/100))
    #imgMean=np.mean(noisy)
    #filtered = cv2.bilateralFilter(noisy, kernel[0], imgMean, diagonal, None,cv2.BORDER_DEFAULT, )
    
   # filtered = AdaptiveLocalFilter(noisy)
    runTime += time.time() - startTime
    mse += MSE(image, filtered)
    psnr += PSNR(image, filtered)

mse=mse/12
psnr=psnr/12
runTime=runTime/12
print(mse)
print(psnr)
print(runTime)

# ------------------------- TESTING ------------------------------

"""



# MAIN
kernel=(3,3)
noise_Probability=noise_high
image = cv2.imread('./dataset/30.jpg',cv2.IMREAD_GRAYSCALE)
#NOISE
noisy=image;
#noisy = add_g_noise(image,1000)
#noisy = add_sp_noise(image, noise_Probability)
#
showImg(image, 'Original')
showImg(noisy, 'With Noise')
cv2.imwrite("original.jpg",image,(128,128))
cv2.imwrite("noisy.jpg",noisy,(128,128))
filtered=np.zeros(image.shape,np.uint8)
cv2.waitKey(0)
# Box Filter
startTime = time.time()
filtered = cv2.boxFilter(noisy, -1, kernel, None, (-1, -1), 1, cv2.BORDER_DEFAULT)#------ FILTER
runTime=time.time()-startTime
mse=MSE(image,filtered)
psnr=PSNR(image,filtered)
showImg(filtered,"Box Filter / Kernel:"+str(kernel[0])+"x"+str(kernel[1]))
print("\n##########################################################################\n")
print("Box Filter With Kernel("+str(kernel[0])+"x"+str(kernel[1])+") :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.imwrite("filtered.jpg",filtered,(128,128))
cv2.waitKey(0)
# Gaussian
startTime = time.time()
filtered = cv2.GaussianBlur(noisy,kernel,3,None,3) # FILTER (image,kernel size,sigmax,-,sigmay)
runTime=time.time()-startTime
mse=MSE(image,filtered)
psnr=PSNR(image,filtered)
showImg(filtered,"Gaussian Filter / Kernel:"+str(kernel[0])+"x"+str(kernel[1]))
print("\n##########################################################################\n")
print("Gaussian Filter With Kernel("+str(kernel[0])+"x"+str(kernel[1])+") :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.imwrite("filtered.jpg",filtered,(128,128))
cv2.waitKey(0)
# Adaptive median
startTime = time.time()
AdaptiveMedianFilter(noisy,filtered,7)#------ (image,dest,maxK=7)
runTime=time.time()-startTime
mse=MSE(image,filtered)
psnr=PSNR(image,filtered)
showImg(filtered,"Adaptive Median Filter With Max Kernel(7x7) :")
print("\n##########################################################################\n")
print("Adaptive Median Filter With Max Kernel(7x7) :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.waitKey(0)
# Median
startTime = time.time()
filtered = cv2.medianBlur(noisy,kernel[0])# FILTER  (image,kernel size)
runTime=time.time()-startTime
mse=MSE(image,filtered)
psnr=PSNR(image,filtered)
showImg(filtered,"Median Filter / Kernel:"+str(kernel[0])+"x"+str(kernel[1]))
print("\n##########################################################################\n")
print("Median Filter With Kernel("+str(kernel[0])+"x"+str(kernel[1])+") :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.waitKey(0)
# Bilaterlar
startTime = time.time()
filtered = cv2.bilateralFilter(noisy, kernel[0], 50, 15,None,cv2.BORDER_DEFAULT,) #(image,kernel,sigmaRange,sigmaSpace)
runTime=time.time()-startTime
mse=MSE(noisy,filtered)
psnr=PSNR(noisy,filtered)
showImg(filtered,"Bilateral Filter / Kernel:"+str(kernel[0])+"x"+str(kernel[1]))
print("\n##########################################################################\n")
print("Bilateral Filter With Kernel("+str(kernel[0])+"x"+str(kernel[1])+") :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.waitKey(0)
# LOCAL Filter
startTime = time.time()
filtered = AdaptiveLocalFilter(noisy)#------ FILTER
runTime=time.time()-startTime
mse=MSE(image,filtered)
psnr=PSNR(image,filtered)
showImg(filtered,"Adaptive Local Filter / Kernel:"+str(kernel[0])+"x"+str(kernel[1]))
print("\n##########################################################################\n")
print("Adaptive Local Filter With Kernel("+str(kernel[0])+"x"+str(kernel[1])+") :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.waitKey(0)
# LOCAL Filter
startTime = time.time()
filtered = AdaptiveLocalFilter(noisy)#------ (image,kernel=3)
runTime=time.time()-startTime
mse=MSE(image,filtered)
psnr=PSNR(image,filtered)
showImg(filtered,"Adaptive Local Filter / Kernel:"+str(kernel[0])+"x"+str(kernel[1]))
print("\n##########################################################################\n")
print("Adaptive Local Filter With Kernel("+str(kernel[0])+"x"+str(kernel[1])+") :")
print('*MSE= '+str(mse)+" *PSNR= "+str(psnr)+" *RunTime= "+str(runTime))
cv2.waitKey(0)
#
cv2.waitKey(0)
cv2.destroyAllWindows()





