

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
        return 'Speed limit(20km/hr)'
    elif classNo == 1:
        return 'Speed limit(30km/hr)'
    elif classNo == 2:
        return 'Speed limit(50km/hr)'
    elif classNo == 3:
        return 'Speed limit(60km/hr)'
    elif classNo == 4:
        return 'Speed limit(70km/hr)'
    elif classNo == 5:
        return 'Speed limit(80km/hr)'
    elif classNo == 6:
        return ' End of Speed limit(80km/hr)'
    elif classNo == 7:
        return 'Speed limit(100km/hr)'
    elif classNo == 8:
        return 'Speed limit(120km/hr)'
    elif classNo == 9:
        return 'Nopassing'
    elif classNo == 10:
        return 'No traffic sign found'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return '.'
    elif classNo == 13:
        return 'No traffic sign found'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vehicles'
    elif classNo == 16:
        return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the left'
    elif classNo == 21:
        return 'Double curve '
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road Work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children Crossing'
    elif classNo == 29:
        return  'Bicycles Crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Round about mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vehicles over 3.5 metric tons'
    elif classNo == 43:
        return 'unknown'


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
        speak = wincom.Dispatch("SAPI.SpVoice")
        text=getCalssName(classIndex)
        speak.Speak(text)
        #time.sleep(.5)
        

        
       
            
 
         
    if cv2.waitKey(1) and 0xFF == ord('q'):
    
        break