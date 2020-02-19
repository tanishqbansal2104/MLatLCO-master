import cv2 as cv 
import numpy as np

#taking camera resourse from camera 

cam = cv.VideoCapture(0)

#Dectector
dectector=cv.CascadeClassifier("haar_frontal.xml")
def main():
    while True:
        #Reading the frames
        _,frame=cam.read()
        #frame=cv.resize(frame,None,0.8,0.8,cv.INTER_LINEAR)
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        #detect the face
        faces = dectector.detectMultiScale(gray,1.3,5)

        for face in faces:
            x,y,w,h = face
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

        #show the frames
        cv.imshow("Face Dectector",frame)
        key=cv.waitKey(1)

        if key & 0xff == ord('q'):
            cv.destroyAllWindows()
            break
    cam.release()
if __name__=="__main__":
    main()

