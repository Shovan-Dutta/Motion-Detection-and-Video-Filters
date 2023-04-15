import cv2 as cv


camera = cv.VideoCapture(0)

_, frame1 = camera.read()
_, frame2 = camera.read()


while True:

    #live video motion detection using contours
    diff = cv.absdiff(frame1, frame2) #Calculates the per-element absolute difference between two frames
    gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY) #converts the difference frame to gray image
    blur = cv.GaussianBlur(gray,(5,5),0) #to reduce noise in the difference frame
    _, thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY) # getting a binary image
    dilated  = cv.dilate (thresh, None, iterations = 3) # determines the shape of a pixel neighborhood 
    contours, _ = cv.findContours(dilated,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #The function retrieves contours from the binary image

    for contour in contours:
        if cv.contourArea(contour)>200:
            cv.drawContours(frame1, contours, -1, (0,0,255),2) # highlights the movement area
            cv.putText(frame1,'Status: {}'.format('Movement'),(10,40),cv.FONT_ITALIC,1,(0,0,255), 3)
        cv.imshow('Camera',frame1)
    frame1 = frame2
    _, frame2 = camera.read()

    if cv.waitKey(5) == ord("x"):
        break

camera.release()
cv.destroyAllWindows()
