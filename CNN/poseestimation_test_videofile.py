
#https://github.com/nicknochnack/MediaPipePoseEstimation


import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



     

cap = cv2.VideoCapture("//home/arsal/Desktop/MLproject/Data/normal/1.webm")

# Curl counter variables
counter = 0 
stage = None

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame=cv2.resize(frame, [640,480])
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
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
            
            
            # Visualize angle
            cv2.putText(image, str(angleLeftelbow.astype(int)), 
                           tuple(np.multiply(Lelbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
                                
            cv2.putText(image, str(angleLeftshoulder.astype(int)), 
                           tuple(np.multiply(Lshoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(angleLefthip.astype(int)), 
                           tuple(np.multiply(Lhip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(angleLeftknee.astype(int)), 
                           tuple(np.multiply(Lknee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angleRightelbow.astype(int)), 
                           tuple(np.multiply(Relbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                                )
                                
            cv2.putText(image, str(angleRightshoulder.astype(int)), 
                           tuple(np.multiply(Rshoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(angleRighthip.astype(int)), 
                           tuple(np.multiply(Rhip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(angleRightknee.astype(int)), 
                           tuple(np.multiply(Rknee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                                )
                       
        except:
            pass
        
        
        # Render detections
#        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

