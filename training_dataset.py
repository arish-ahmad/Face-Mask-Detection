import cv2
import numpy as np
#img=cv2.imread('sample2.jpg')
haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            #img[rows,column,color channel] 3d array
            face_slice=img[y:y+h,x:x+w,:]
            face_slice=cv2.resize(face_slice,(50,50))
            print(len(data))
            if len(data)<1001:
                print(face_slice)
                data.append(face_slice)

        cv2.imshow('pic',img)
        if cv2.waitKey(1)==27 or len(data) >=1000:
            break
    else:
        print('camera is not working')
np.save('withoutmask.npy',data)
capture.release()   
cv2.destroyAllWindows()