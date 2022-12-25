
#https://github.com/nicknochnack/MediaPipePoseEstimation


import cv2
import mediapipe as mp
import numpy as np
import os

#print (len(os.listdir("//home/arsal/Desktop/MLproject/Data/normal/")))

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
open('foo2.csv', 'w').close() #clear file
f=open('foo2.csv','a') #open file in append mode

label=0

total_normal=len(os.listdir("//home/arsal/Desktop/MLproject/Data/normal/"))
total_blind=len(os.listdir("//home/arsal/Desktop/MLproject/Data/blind/"))
for i in range(total_normal+total_blind):
        if(i<int(total_normal)):
                label=0
                source="//home/arsal/Desktop/MLproject/Data/normal/" + str(i+1) +".mp4"
        else:
                label=1
                source="//home/arsal/Desktop/MLproject/Data/blind/" + str(int(i+1-(total_normal))) +".mp4"
          

        cap = cv2.VideoCapture(source)
        #cap = cv2.VideoCapture("/home/arsal/Desktop/MLproject/Data/blind/6.mp4")
        print (source)
        # Curl counter variables
        counter = 0 
        stage = None

        dataset=np.array([np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,label])




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
                    
                    dataset=np.vstack((dataset,[cap.get(cv2.CAP_PROP_POS_MSEC),angleLeftelbow,angleLeftshoulder,angleLefthip,angleLeftknee,angleRightelbow,angleRightshoulder,angleRighthip,angleRightknee,label]))
                               
                except:
                    dataset=np.vstack((dataset,[cap.get(cv2.CAP_PROP_POS_MSEC),angleLeftelbow,angleLeftshoulder,angleLefthip,angleLeftknee,angleRightelbow,angleRightshoulder,angleRighthip,angleRightknee,label]))
                    pass
                
                
            cap.release()
            cv2.destroyAllWindows()
        
        np.savetxt(f, dataset, delimiter=",")

f.close()
 

