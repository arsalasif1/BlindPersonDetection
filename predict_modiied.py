import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## to ignore fazool warnings
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import numpy as np
from sklearn.preprocessing import normalize
import time
import os
import csv
import os
import numpy as np
import cv2
import mediapipe as mp
num_coords=501
dataset=np.zeros(132)
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
landmarks = ['class']
#for val in range(1, num_coords+1):
#    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

#total_normal=len(os.listdir("/home/arsal/Desktop/MLproject/Data2/normal/"))
#total_blind=len(os.listdir("/home/arsal/Desktop/MLproject/Data2/blind/"))
#with open('coords.csv', mode='w', newline='') as f:
#    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    csv_writer.writerow(landmarks)

for i in range(10):
#    if(i<int(total_normal)):
#        class_name = "Normal"
#        source="/home/arsal/Desktop/MLproject/Data2/normal/" + str(i+1) +".mp4"
#    else:
#        class_name = "Blind"
#        source="/home/arsal/Desktop/MLproject/Data2/blind/" + str(int(i+1-(total_normal))) +".mp4"



#    print(i)
#    cap = cv2.VideoCapture('/home/arsal/Desktop/MLproject/Data/14.mp4')
#    cap = cv2.VideoCapture("/home/arsal/Desktop/MLproject/Data2/normal/2.mp4")
    source="/home/arsal/Desktop/MLproject/Data/crossvalidation/normal/"+str(i)+ ".mp4"
    print(source)    
    cap = cv2.VideoCapture(source)
    
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        while cap.isOpened():
            ret, frame = cap.read()
            if ret==0:
                break
            frame=cv2.resize(frame, [640,480])
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
        
            # Make Detections
            results = holistic.process(image)

        
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            

            #  Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
                        

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            
                # Concate rows
                row = pose_row  #+face_row
                row = np.array([row])
                dataset=np.vstack((dataset,row))
#                print(dataset.shape)
                # Append class name 
#                row.insert(0, class_name)
            
                # Export to CSV
#                with open('coords.csv', mode='a', newline='') as f:
#                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                    csv_writer.writerow(row) 
            
            except:
                pass
                                
           # cv2.imshow('Raw Webcam Feed', image)
          #  if cv2.waitKey(10) & 0xFF == ord('q'):
           #     break

    cap.release()
    cv2.destroyAllWindows()





## load the dataset
##dataset = loadtxt('normal.csv', delimiter=',')
## split into input (X) and output (y) variables
##dataset=dataset[~np.isnan(dataset).any(axis=1)]

#        try:
#                k=np.argwhere(np.isnan(dataset))[:,0]      
#                dataset=np.delete(dataset, (k,k-1,k-2,k-3,k-4,k-5,k-6,k-7,k-8,k-9,k-10,k-11,k-12,k-13,k-14,k-15), axis = 0)
#                dataset=np.delete(dataset, (k,k+1,k+2,k+3,k+4,k+5,k+6,k+7,k+8,k+9,k+10,k+11,k+12,k+13,k+14,k+15), axis = 0)
#        except: 
#                pass



#        dataset = normalize(dataset, axis=0, norm='max')
        #X = np.hstack([dataset[:,0:3],dataset[:,5:7]])
    X=dataset
# define the keras model
    model = keras.models.load_model('model')
# make class predictions with the model
    predictions = model.predict(X)
# summarize the first 5 cases
#for i in range(len(X)):
#	print('%s => %d' % (X[i].tolist(), predictions[i]))
#print (predictions)
    print ("confidence of Blind detection")
    print ((np.sum(predictions,axis=0)/len(predictions)*100))
    if((np.sum(predictions,axis=0)/len(predictions)*100)>10):
      print('blind')
    else:
      print('normal')
        

