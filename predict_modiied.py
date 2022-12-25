import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## to ignore fazool warnings
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import numpy as np
from sklearn.preprocessing import normalize

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

     
for i in range(1):

#        cap = cv2.VideoCapture("/home/arsal/Desktop/MLproject/Data/crossvalidation/normal.webm")
        cap = cv2.VideoCapture("/home/arsal/Desktop/MLproject/Data/crossvalidation2/9.mp4")

        # Curl counter variables
        counter = 0 
        stage = None

        dataset=np.array([np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])




        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret==0:
                        break
                frame=cv2.resize(frame, [640,480])

                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
              
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #time references
        #        print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #        print(cap.get(cv2.CAP_PROP_POS_MSEC))
        #        dataset=np.vstack((dataset,[cap.get(cv2.CAP_PROP_POS_MSEC),5]))
                
        #        print(dataset)
                #initilize with nan 
                angleLeftelbow = np.nan
                angleLeftshoulder = np.nan
                angleLefthip = np.nan
                angleLeftknee = np.nan
                angleRightelbow = np.nan
                angleRightshoulder = np.nan
                angleRighthip = np.nan
                angleRightknee = np.nan
                
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    #initilize with nan 
                    angleLeftelbow = np.nan
                    angleLeftshoulder = np.nan
                    angleLefthip = np.nan
                    angleLeftknee = np.nan
                    angleRightelbow = np.nan
                    angleRightshoulder = np.nan
                    angleRighthip = np.nan
                    angleRightknee = np.nan
                    
                    # Calculate angle Left elbow
                    Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angleLeftelbow = calculate_angle(Lshoulder, Lelbow, Lwrist)
                    
                    # Calculate angle Left Shoulder
                    Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    angleLeftshoulder = calculate_angle(Lhip,Lshoulder, Lelbow)
                    
                    # Calculate angle Left Hip
                    Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angleLefthip = calculate_angle(Lknee,Lhip, Lshoulder)
                    
                    # Calculate angle Left Knee
                    Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angleLeftknee = calculate_angle(Lankle, Lknee, Lhip)
                    
                    # Calculate angle Right elbow
                    Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angleRightelbow = calculate_angle(Rshoulder, Relbow, Rwrist)
                    
                    # Calculate angle Right Shoulder
                    Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    angleRightshoulder = calculate_angle(Rhip,Rshoulder, Relbow)
                    
                    # Calculate angle Right Hip
                    Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    angleRighthip = calculate_angle(Rknee,Rhip, Rshoulder)
                  
                    # Calculate angle Right Knee
                    Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    angleRightknee = calculate_angle(Rankle, Rknee, Rhip)
                    
                    dataset=np.vstack((dataset,[cap.get(cv2.CAP_PROP_POS_MSEC),angleLeftelbow,angleLeftshoulder,angleLefthip,angleLeftknee,angleRightelbow,angleRightshoulder,angleRighthip,angleRightknee]))
                               
                except:
                    dataset=np.vstack((dataset,[cap.get(cv2.CAP_PROP_POS_MSEC),angleLeftelbow,angleLeftshoulder,angleLefthip,angleLeftknee,angleRightelbow,angleRightshoulder,angleRighthip,angleRightknee]))
                    pass
                
                
            cap.release()
            cv2.destroyAllWindows()





# load the dataset
#dataset = loadtxt('normal.csv', delimiter=',')
# split into input (X) and output (y) variables
#dataset=dataset[~np.isnan(dataset).any(axis=1)]

try:
        k=np.argwhere(np.isnan(dataset))[:,0]      
        dataset=np.delete(dataset, (k,k-1,k-2,k-3,k-4,k-5,k-6,k-7,k-8,k-9,k-10,k-11,k-12,k-13,k-14,k-15), axis = 0)
        dataset=np.delete(dataset, (k,k+1,k+2,k+3,k+4,k+5,k+6,k+7,k+8,k+9,k+10,k+11,k+12,k+13,k+14,k+15), axis = 0)
except: 
        pass



dataset = normalize(dataset, axis=0, norm='max')
#X = np.hstack([dataset[:,0:3],dataset[:,5:7]])
X=dataset[:,0:9]
# define the keras model
model = keras.models.load_model('/home/arsal/Desktop/MLproject/Model')
# make class predictions with the model
predictions = model.predict(X)
# summarize the first 5 cases
#for i in range(len(X)):
#	print('%s => %d' % (X[i].tolist(), predictions[i]))
#print (predictions)
print ("confidence of Blind detection")
print (np.sum(predictions,axis=0)/len(predictions)*100)
