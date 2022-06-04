import cv2
cap = cv2.VideoCapture(0)
#url= "http://192.168.10.27:8080/video"
#cap.open(url)
while cap.isOpened():
    ret,frame=cap.read()
    cv2.imshow("WebCam",frame)
