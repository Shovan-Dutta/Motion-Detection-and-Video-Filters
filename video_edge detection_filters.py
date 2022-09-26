import cv2 as cv
import numpy as np

camera  =cv.VideoCapture(0)

while True:
    ret,frame = camera.read()
    if ret == False:
        break
    #getting width and height of image
    height,width, _ = frame.shape

    #creating copy of image using resize function
    resizedImage = cv.resize(frame,(width,height),interpolation = cv.INTER_AREA)
    
    #creating 3*3 filter, inorder to sharpenthe image/frame
    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])
    
    #Applying kernel on the frame using filter 2D function.
    sharpenImage = cv.filter2D(resizedImage,-1,kernel)

    #converting image into Grayscale image.
    gray = cv.cvtColor(sharpenImage,cv.COLOR_BGR2GRAY)

    #creating inverse of sharpen image.
    inverseImage = 255-gray

    #applying Gaussian blur on the image.
    bluredImage = cv.GaussianBlur(inverseImage, (15,15),0,0)

    #create a pencilSketch using divide function on opencv.
    pencilSketch = cv.divide(gray, 255-bluredImage, scale=256)

    #show the frame on the screen.
    cv.imshow('Camera',frame)
    cv.imshow("sketch", pencilSketch)

    #live video edge detectation using canny filter (usually less noisy image depends in the threshold)
    edges = cv.Canny(frame,100,100)
    cv.imshow('Canny', edges)

    #live video edge detectation using laplacian filter(usually very noisy image )
    laplacian = cv.Laplacian(frame,cv.CV_64F)
    laplacian = np.uint8(laplacian)
    cv.imshow('Laplacian',laplacian)

    key = cv.waitKey(1)
    #to quit the program
    if key == ord('x'):
        break

cv.destroyAllWindows()
camera.release()