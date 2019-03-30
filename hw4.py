#!/usr/bin/env python
# coding: utf-8

# In[63]:


from PIL import Image, ImageDraw 
import numpy as np
import math
import imageio
from copy import deepcopy
import cv2

def to_integral_image(img_arr):
    """
    Calculates the integral image based on this instance's original image data.
    """
    row_sum = np.zeros(img_arr.shape)
    # we need an additional column and row of padding zeros
    integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
    for x in range(img_arr.shape[1]):
        for y in range(img_arr.shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
            integral_image_arr[y+1, x+1] = integral_image_arr[y+1, x-1+1] + row_sum[y, x]
    return integral_image_arr


def sum_region(integral_img_arr, top_left, bottom_right):
    """
    Calculates the sum in the rectangle specified by the given tuples.
    """
    top_left=(top_left[0]-1,top_left[1]-1)
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return integral_img_arr[bottom_right] - integral_img_arr[top_right] - integral_img_arr[bottom_left] + integral_img_arr[top_left]


class HaarFeature(object):
    """
    Class representing a haar-like feature.
    """

    def __init__(self, feature_type, top_left, bottom_right, threshold, polarity, error, weight, flag):
        """
        Creates a new haar-like feature with relevant attributes.
        """
        self.type = feature_type
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.width = bottom_right[0]-top_left[0]
        self.height = bottom_right[1]-top_left[1]
        self.threshold = threshold
        self.polarity = polarity
        self.error=error
        self.weight=weight
        self.flag=flag
    
    def get_score(self, int_img):
        """
        Get score for given integral image array.
        """
        score = 0
        if self.type == (1,2):
            first = sum_region(int_img, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
            score = first - second
        elif self.type == (2,1):
            first = sum_region(int_img, self.top_left, (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]), self.bottom_right)
            score = first - second
        return score
    
    def get_vote(self, int_img):
        """
        Get vote of this feature for given integral image, the vote is 1 or -1.
        """
        score = self.get_score(int_img)
        return 1 if self.polarity * (score-self.threshold) >= 0 else -1

#helper function to sum positive numbers in an array
def sum_positive(array):
    s=0
    l=len(array)
    for i in range(l):
        if array[i]>0:
            s=s+array[i]
    return s
#helper function to sum negative numbers in an array
def sum_negative(array):
    s=0
    l=len(array)
    for i in range(l):
        if array[i]<0:
            s=s+array[i]
    return s 
    
#given an array of lables and weights of each image (label), find the threshold for this weaker learner
def find_threshold(array, weights):
    index=1
    p=1
    l=len(array)
    output_error=1
    temp=np.multiply(array,weights)
    Lp=0
    Ln=0
    Rp=sum_positive(temp)
    Rn=sum_negative(temp)
    #try every index
    for i in range(1,l):
        t=temp[i]
        if t>0:    
            Lp=Lp+t      
            Rp=Rp-t
        else:
            Ln=Ln+t
            Rn=Rn-t
        error=min(Lp+abs(Rn),abs(Ln)+Rp)
        if error < output_error:
            output_error=error
            index = i
            if Lp+abs(Rn) < Rp+abs(Ln):
                p=1
            else:
                p=-1
    #return the best polarity, the index of the image (whose score will be the threshold),
    #and the error of this weak learner
    return (index,p,output_error)          

def learn(features, images, labels, weights):
    """
    This is the mean funtion we use, every time we feed the images, labels, features,
    and weigts of the images, it will output the current best weaklearner and set the
    parameters.
    """
    # select classifiers
    lenf = len(features)
    leni = len(labels)
    fi=0
    min_error=1
    for i in range(lenf):
        temp=np.zeros((leni,3))
        for j in range(leni):
            img=images[j]
            x=features[i].get_score(img)
            y=labels[j]
            temp[j][0]=x
            temp[j][1]=y
            temp[j][2]=weights[j]
        temp=temp[temp[:,0].argsort()]
        #get the labels and weights we need to find the threshold for this feature
        tup=find_threshold(temp[:,1],temp[:,2])
        index=tup[0]
        features[i].threshold=temp[index][0]
        features[i].polarity=tup[1]        
        error=tup[2]
        #to record the best feature 
        if (error < min_error) and (features[i].flag==0):
            min_error=error
            fi=i  

    # already find the best feature, update its parameters
    # flag indicates whether this feature has already been picked before
    features[fi].flag=1
    features[fi].error=min_error
    # find the weight of this chosen weak learner
    if min_error>0:
        z=2*(min_error*(1-min_error)) ** (1/2)
        a = 0.5 * np.log((1 - min_error) / min_error)
    else:
        a=2
        z=0
    features[fi].weight=a 
    # update the weights of the data (images)
    if z!=0:
        for i in range(leni):
            vote=features[i].get_vote(images[i])
            weights[i]=weights[i]*math.exp(-a*labels[i]*vote)/z
    # normalize the weights
    s=np.sum(weights)
    weights=weights/s
    return (fi,weights)

# This generates all the features we need
all_features=[]
for i in range(1,65,3):
    for j in range(1,65,3):
        m1=min(i+16,65)
        m2=min(j+16,65)
        for k1 in range(i,m1,4):
            for h1 in range(j+3,m2,4):
                f = HaarFeature((1,2),(i,j),(k1,h1),0,1,0,0,0) 
                all_features.append(f)
        for k2 in range(i+3,m1,4): 
            for h2 in range(j,m2,4):
                f = HaarFeature((2,1),(i,j),(k2,h2),0,1,0,0,0)
                all_features.append(f) 

# given a classifier, this function outputs whether an image is a face or not according to this classifier
def test(img, classifiers, threshold=3):
    l=len(classifiers)
    s=0
    for i in range(l):
        s=s+classifiers[i].weight*classifiers[i].get_vote(img) 
    if s>=threshold:
        return (1,s)
    return (-1,s)

# load the images
images=[]
labels=[]
for i in range (1000):
    j=i
    j=str(j)
    im = imageio.imread('Downloads/faces/face'+j+'.jpg')
    im=np.array(im).mean(axis=2)
    im=to_integral_image(im)
    images.append(im)
    labels.append(1)
for i in range (1000):
    j=i
    j=str(j)
    im = imageio.imread('Downloads/background/'+j+'.jpg')
    im=np.array(im).mean(axis=2)
    im=to_integral_image(im)
    images.append(im)
    labels.append(-1)

cascade=[]
thres=[]
iter=0
for j in range(6):    
    classifiers=[]
    for i in range(5):
        l=len(images)
        weights=np.full(l,1/l)
        t = learn(all_features, images, labels, weights)
        p=t[0]
        weights=t[1]
        classifiers.append(all_features[p])
    l=len(images)
    mini=0
    # set the capital Theta to make sure we correctly classify all faces
    for i in range(l):
        t=test(images[i],classifiers)
        if labels[i]==1:
            if t[1]<mini:
                mini=t[1]
    thres.append(deepcopy(mini))
    cascade.append(deepcopy(classifiers))
    # elimiate the images we correctly classify as backgrounds
    images_temp=[]
    labels_temp=[]
    for i in range(l):
        t=test(images[i],classifiers,mini)
        if (labels[i]!=t[0]) or (labels[i]==1):
            images_temp.append(images[i])
            labels_temp.append(labels[i])
    images=deepcopy(images_temp)
    labels=deepcopy(labels_temp)
    # deal with the situation is before we use up all 5 classifiers, the error has already been 0
    iter=iter+1
    if len(images)<2:
        break

    
def get_v(cascade, int_img):
        """
        helper funtion to avoid overlapping of red patches detecting faces 
        """
        l=len(cascade)
        s=0
        for i in range(l):
            score = cascade[i].get_score(int_img)
            ab=abs(cascade[i].polarity * (score-cascade[i].threshold))
            s=s+cascade[i].weight*ab
        return s    

#slide our cascade to detect faces in the test image   
coor=np.zeros((1280,1600))
im1=Image.open('Downloads/test_img.jpg')
#add chanels to the black&white test image, so we can draw "red sqaures" on it
im1=cv2.cvtColor(np.array(im1),cv2.COLOR_GRAY2RGB)
img=Image.fromarray(im1, 'RGB')
im2=Image.open('Downloads/test_img.jpg')
imay=np.array(im2)
draw=ImageDraw.Draw(img)
# determine where we shoud place our pateches
for i in range(0,1216,4):
    for j in range(0,1536,4):
        y=to_integral_image(imay[i:i+64,j:j+64])
        flag=0
        for k in range(iter):  
            t=test(y,cascade[k])
            if t[0]==1:
                flag=flag+1
        if (flag>=6):
            v=0
            for k in range(iter):  
                v=v+get_v(cascade[k],y)
            coor[i,j]=v
# avoid overlapping
for i in range(0,1216,4):
    for j in range(0,1536,4):
        ff=1
        for z1 in range(-7,8):
            for z2 in range(-7,8):     
                if coor[i,j]<coor[i-4*z1,j-4*z2]:
                    ff=0
        if (ff==1) and coor[i,j]>0: 
            draw.line([(j,i),(j,i+64)],fill='red')
            draw.line([(j,i),(j+64,i)],fill='red')
            draw.line([(j+64,i),(j+64,i+64)],fill='red')
            draw.line([(j,i+64),(j+64,i+64)],fill='red')            
img.save('Downloads/result4.jpg')         

