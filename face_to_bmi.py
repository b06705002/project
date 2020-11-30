import face_recognition
import cv2
import dlib


from keras.models import load_model
from joblib import load

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import numpy as np

def cossMulti(v1, v2):

    return v1[0]*v2[1] - v1[1]*v2[0]


def getFaceArea(polygon):

    n = len(polygon)
    if n < 3:
        return -1
    
    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]

    area = 0
    for i in range(1, n):
        area = area + cossMulti(vectors[i-1, :], vectors[i, :]) / 2

    return area


def faceArea(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rects = detector(img_gray, 0)
    faceArea = 0
    centerArea = 0
    
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()]) 
        landmarks = np.array(landmarks)
        facePoints = np.array([landmarks[0], landmarks[16], landmarks[12], landmarks[8], landmarks[4]])
        faceArea = getFaceArea(facePoints)
        centerPoints = np.array([landmarks[36], landmarks[45], landmarks[54], landmarks[48]])
        centerArea = getFaceArea(centerPoints)
        
    return faceArea, centerArea

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



def getPointsDistance(x1, y1, x2, y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


def faceDistance(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rects = detector(img_gray, 0)
    chinDis = 0
    eyesDis = 0
    
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()]) 
        landmarks = np.array(landmarks)
        chinDis = getPointsDistance(landmarks[4][0], landmarks[4][1], landmarks[12][0], landmarks[12][1])
        leftEyesX, leftEyesY = getEyesCenter(landmarks[37], landmarks[38], landmarks[40], landmarks[41])
        rightEyesX, rightEyesY = getEyesCenter(landmarks[43], landmarks[44], landmarks[46], landmarks[48])
        eyesDis = getPointsDistance(leftEyesX, leftEyesY, rightEyesX, rightEyesY)
    return chinDis, eyesDis

def get_face_encoding(image_path):
    print(image_path, end = "")
    try:  
        picture_of_me = face_recognition.load_image_file(image_path)
        my_face_encoding = face_recognition.face_encodings(picture_of_me)
        if not my_face_encoding:
            print("  =====>  No face found!!!")
            return np.zeros(128).tolist(), 0
        print("  =====>  OK!!!")
        return my_face_encoding[0].tolist(), 1
    except(OSError, NameError):
        print('  =====>  OSError, Path:',image_path)
        return np.zeros(128).tolist(), 0

def face_bmi_test(image_path):
    
    model_c = load_model('model.h5')

    image = image_path
    face_enc, comf = get_face_encoding(image)
    if comf == 1:
        chins, eyes = faceDistance(image)
        face_area, center_area = faceArea(image)
        if (chins * eyes) != 0 and (face_area * center_area) != 0:
            dis = eyes/chins
            are = face_area/center_area
        else:
            print("Cannot get features!")
    else:
        print("Cannot get image!")
        
    face_enc.append(dis)
    face_enc.append(are)
    face_enc = np.array(face_enc).reshape(1, -1)
    scaler = load('std_scaler.bin')
    face_enc = scaler.transform(face_enc)

    img = mpimg.imread(image)
    plt.imshow(img) 
    plt.axis('off') 
    plt.show()
    print(model_c.predict(face_enc).argmax())
    
    if model_c.predict(face_enc).argmax() == 0:
        return("Underweight")
    elif model_c.predict(face_enc).argmax() == 1:
        return("Normalweight")
    elif model_c.predict(face_enc).argmax() == 2:
        return("Pre-obesity")
    else:
        return("Obesity")