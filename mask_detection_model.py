import cv2
import numpy as np
from sklearn.svm import SVC   
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 

#load training dataset for model training
with_mask=np.load('withmask.npy',allow_pickle=True)
without_mask=np.load('withoutmask.npy',allow_pickle=True)

#400 images with mask
with_mask=with_mask.reshape(with_mask.shape[0],50*50*3)

#400 image without mask    
without_mask=without_mask.reshape(without_mask.shape[0],50*50*3)

#concatenate both array in single array
x=np.r_[with_mask,without_mask] #x variable 800 images

# 0 for wearing mask
y=np.zeros(x.shape[0])

#1.0 or 1 for not wearing mask
y[x.shape[0]//2:]=1.0

#dividing 25% data into testing and 75% into training
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

#reducing column using principal component analysis
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)

# using support vector machine and support vector classification pridict the scenario
svm=SVC()
svm.fit(x_train,y_train)

#again transforming 2d into 3d
x_test=pca.transform(x_test)
y_prediction=svm.predict(x_test)

#comparing actual testing data with prediction
accur=accuracy_score(y_test,y_prediction)


# loop over the frames from the video stream
var={0:'Mask',1:'NO mask'}
haar_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
mask_value,without_mask_value=0,0
while True:
    flag,img=capture.read()     # flag is true if camera is working properly
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            # grab the frame from the threaded video stream and resize it with face area
            face_slice=img[y:y+h,x:x+w,:]
            face_slice=cv2.resize(face_slice,(50,50))
            face_slice=face_slice.reshape(1,-1)

            # get the probability from model
            face_slice=pca.transform(face_slice)
            predict=svm.predict(face_slice)
            n=var[int(predict)]
            # set green color for wearing mask otherwise set red color 
            if n == 'Mask':
                color=(0, 255, 0)  #green
                mask_value+=1 
            else:
                color=(0, 0, 255)  #red
                without_mask_value+=1

            # display  and bounding box rectangle on the output
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,n,(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.45,color,2)
            

        # show the output frame
        cv2.imshow('pic',img)

        #ASCII value of Escape =27
        if cv2.waitKey(1)==27:
            break
    else:
        print('camera is not working')
capture.release()   
cv2.destroyAllWindows()
print(f'\nAccuracy Rate: {accur}\nmask time: {mask_value} \nwithout mask time: {without_mask_value}')

# x-coordinates of left sides of bars 
Categories = [1, 2] 

# heights of bars 
frames= [mask_value, without_mask_value] 

# labels for bars 
tick_label = ['Mask ','No mask'] 

# plotting a bar chart 
plt.bar(Categories, frames, tick_label = tick_label, width = 0.5, color = ['green','red'])

# naming the x-axis 
plt.xlabel('Categories') 

# naming the y-axis 
plt.ylabel('Frames per Session') 

# function to show the plot 
plt.show() 




