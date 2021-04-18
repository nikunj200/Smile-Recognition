"""
Step1: Find faces
Step2: Find smiles only in that face
Step3: Show whole frame with smile 
"""


from cv2 import cv2

#first detecting faces
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('haarcascade_smile.xml')
#webcam/video capture (0-webcam)
webcam=cv2.VideoCapture(0)

while True:

    #read frames
    successful_frame_read,frame=webcam.read()
    
    #if error then abort
    if not successful_frame_read:
        break

    #grayscale conversion
    frame_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detect faces
    faces=face_detector.detectMultiScale(frame_grayscale)


    """
    x,y : top left point of face
    w: width
    h: height

    rectangele(frame,dimensions,color,thikness)
    """
    #rectangles for faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (100,200,50),4)

        #using slicing of lists
        sub_face=frame[y:y+h,x:x+w]

        #change color to grayscale of sub_face
        sub_face_grayscale=cv2.cvtColor(sub_face,cv2.COLOR_BGR2GRAY)
        
        #detect smiles in sub face
        smiles=smile_detector.detectMultiScale(sub_face_grayscale,scaleFactor=1.7,minNeighbors=20)
        
        #lablel
        if len(smiles)>0:
            cv2.putText(frame,'Smile',(x+w-10,y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
        #making rectangle as sub face and then detecting smiles (optimize)
        for (x_s,y_s,w_s,h_s) in smiles:
            cv2.rectangle(sub_face , (x_s , y_s) , (x_s + w_s , y_s + h_s), (50,50,200),4)

    #show frame
    cv2.imshow('Smile Detector',frame)

    #every 1ms waitkey to make realtime
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
