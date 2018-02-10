# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt  
# iscrtavanje slika i plotova unutar samog browsera
#%matplotlib inline 

import matplotlib.pylab as pylab
# prikaz vecih slika 
pylab.rcParams['figure.figsize'] = 16,12

import numpy as np
import cv2 # OpenCV biblioteka



def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=3)

def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=2)


def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)
    
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1
    
    return (x, y)


img = cv2.imread('C:\Users\Jelena\Desktop\SOFT\jag8.jpg')  # ucitavanje slike sa diska
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # konvertovanje iz BGR u RGB model boja (OpenCV ucita sliku kao BGR)

img = cv2.resize(img,(800,600))
cv2.imshow('slika',img)  # prikazivanje slike

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
maska = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([15, 255, 255]))
rezultat_crveno = cv2.bitwise_and(img, img, mask = maska)
cv2.imshow('rez crveno',rezultat_crveno)

img_siva = cv2.cvtColor(rezultat_crveno, cv2.COLOR_RGB2GRAY) # konvert u grayscale
cv2.imshow('gray',img_siva)


img_siva=dilate(img_siva)
img_siva=erode(img_siva)
img_siva=dilate(img_siva)
img_siva=erode(img_siva)
img_siva=dilate(img_siva)
img_siva=erode(img_siva)
img_siva=dilate(img_siva)
img_siva=erode(img_siva)

cv2.imshow('gray2',img_siva)



#razdvajanje jagoda

ret, thresh = cv2.threshold(img_siva,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('thresh',thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.imshow('sure_bg',sure_bg)
cv2.imshow('sure_fg',sure_fg)
cv2.imshow('unknown',unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
cv2.imshow('markers',markers)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
cv2.imshow('img22',img)




img_konture, contours, hierarchy = cv2.findContours(img_siva, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img_konture = rezultat_crveno.copy()
cv2.drawContours(img_konture, contours, -1, (0, 0, 255), 1)

cv2.imshow('sve',img_konture)


contours_jag = [] #ovde ce biti samo konture koje pripadaju jagodama

'''
for contour in contours: # za svaku konturu
    (x,y),radius = cv2.minEnclosingCircle(contour) 
    radius = int(radius)
    center = (int(x),int(y))
   
    
    if radius > 50 and radius < 300 : # uslov da je kontura jagoda
        contours_jag.append(contour)
'''

for contour in contours: # za svaku konturu
    center, size, angle = cv2.minAreaRect(contour) 
    width, height = size
    razlika = width - height
    if width > 80 and width < 500 and height > 80 and height < 500: 
       if  razlika <300: 
        contours_jag.append(contour) #
     #   if width*height >30000:
     #        contours_jag.append(contour) 
        
'''
contours_jag2 = []
for c in contours_jag:
    mx,my,mw,mh = cv2.boundingRect(c)
    topy = int(my)
    boty = int(my + mh)
    leftx = int(mx)
    rightx = int(mx + mw)
    cropped = img_konture[topy:boty, leftx:rightx]
    #cv2.imshow('c',cropped)
    gray = cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)
    #cv2.imshow('g',gray)
    x,y = hist(gray)
    #plt.plot(x, y, 'b')
    #plt.show()
    cl1 = 0
    for i in range(0,5):
        cl1=cl1+y[i]
    if(cl1<(int((mw*mh)*0.8))):
        contours_jag2.append(c)
    
'''
    



img_jagode = rezultat_crveno.copy()
#img_jagode = cv2.circle(img_jagode,center,30,(0,255,0),2)  #nemoj brisati

cv2.drawContours(img_jagode, contours_jag, -1, (0, 0, 255), 1)
#cv2.imshow('img jagode',img_jagode)

#gray = cv2.cvtColor(img_jagode,cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('jgd',img_jagode)
print 'Ova biljka daje %d jagoda' % len(contours_jag)

cv2.waitKey()