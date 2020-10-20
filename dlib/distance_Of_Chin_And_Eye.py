#!/usr/bin/env python
# coding: utf-8

# In[61]:


import cv2
import dlib
import numpy as np


# <strong>get eyes center</strong>

# In[62]:


def getEyesCenter(p1, p2, p3, p4):
    x1 = p1[0]
    x2 = p2[0]
    x3 = p3[0]
    x4 = p4[0]
    y1 = p1[1]
    y2 = p2[1]
    y3 = p3[1]
    y4 = p4[1]
    x = (x1+x2+x3+x4)/4
    y = (y1+y2+y3+y4)/4
    return x, y


# <strong>get points distance</strong>

# In[63]:


def getPointsDistance(x1, y1, x2, y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


# ## get face distance

# In[66]:


def faceDistance(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rects = detector(img_gray, 0)
    
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])  #人臉關鍵點識別
        landmarks = np.array(landmarks)
        chinDis = getPointsDistance(landmarks[4][0], landmarks[4][1], landmarks[12][0], landmarks[12][1])
        leftEyesX, leftEyesY = getEyesCenter(landmarks[37], landmarks[38], landmarks[40], landmarks[41])
        rightEyesX, rightEyesY = getEyesCenter(landmarks[43], landmarks[44], landmarks[46], landmarks[48])
        eyesDis = getPointsDistance(leftEyesX, leftEyesY, rightEyesX, rightEyesY)
    #     for idx, point in enumerate(landmarks):        #enumerate函式遍歷序列中的元素及它們的下標
    #         # 68點的座標
    #         pos = (point[0, 0], point[0, 1])
    #         print(idx,pos)
    #         # 利用cv2.circle給每個特徵點畫一個圈，共68個
    #         cv2.circle(img, pos, 5, color=(0, 255, 0))
    #         # 利用cv2.putText輸出1-68
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         #各引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
    #         cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)
    return chinDis, eyesDis


# In[67]:


# cv2.namedWindow("img", 2)     
# cv2.imshow("img", img)
# cv2.waitKey(0)

