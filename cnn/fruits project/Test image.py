

import numpy as np
import cv2
import pickle
import time
import os
import gtts
from playsound import playsound
import win32com.client as wincom
import time
import cv2


############################################
 

threshold = 0.90         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
# SETUP THE VIDEO CAMERA
#cap = cv2.imread(r"C:\Users\DELL\Desktop\project\zero.jpg", 1)

# IMPORT THE TRANNIED MODEL
pickle_in=open("model_trained.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0:
        return 'Apple'
    elif classNo == 1:
        return 'Avocado'
    elif classNo == 2:
        return 'cheries'
    elif classNo == 3:
        return 'coconut'
    elif classNo == 4:
        return 'grape'
    elif classNo == 5:
        return 'kiwi'
    elif classNo == 6:
        return 'lemon'
    elif classNo == 7:
        return 'Pineapple'
    elif classNo == 8:
        return 'strawberry'
    elif classNo == 9:
        return 'Watermelon'
    


cam=cv2.VideoCapture(0)
while True:
 
    # READ IMAGE
    check,frame=cam.read()
    imgOrignal = frame
     
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
   # cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        #break
        cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        #cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
        print(getCalssName(classIndex))
            # speak = wincom.Dispatch("SAPI.SpVoice")
            # text=getCalssName(classIndex)
            # speak.Speak(text)
            #time.sleep(.5)
            

            
        
            
 
         
    if cv2.waitKey(1) and 0xFF == ord('q'):
    
        break