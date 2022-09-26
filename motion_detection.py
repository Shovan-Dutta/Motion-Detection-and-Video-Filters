import cv2 as cv


camera = cv.VideoCapture(0)

_, frame1 = camera.read()
_, frame2 = camera.read()

while True:

    #live video motion detection using contours
    diff = cv.absdiff(frame1, frame2)
    gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    _, thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
    dilated  = cv.dilate (thresh, None, iterations = 3)
    contours, _ = cv.findContours(dilated,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour)>200:
            cv.drawContours(frame1, contours, -1, (0,0,255),2)
            cv.putText(frame1,'Status: {}'.format('Movement'),(10,40),cv.FONT_ITALIC,1,(0,0,255), 3)
        cv.imshow('Camera',frame1)
    frame1 = frame2
    _, frame2 = camera.read()

    if cv.waitKey(5) == ord("x"):
        break

camera.release()
cv.destroyAllWindows()