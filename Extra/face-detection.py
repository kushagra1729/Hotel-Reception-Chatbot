import cv2
import numpy 
picture = cv2.imread("priyanka.jpg")

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier , opencv-files/lbpcascade_frontalface.xml
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return img[y:y+w, x:x+h], faces[0]

image,x = detect_face(picture)

# now resize the image
WIDTH = 100
HEIGHT = 100
dim = (WIDTH, HEIGHT)

# perform the actual resizing of the image, resized is the resized image. 
resized = cv2.resize(image,(100,100))
cv.imwrite("pc_face.jpg",resize)
# cv2.imshow("Display window",image)
