import cv2

# path to pretrained classifiers xml files
#face_detector_path = "../classifiers/haarcascade_frontalface_default.xml"

knife_detector_path = "../trained_classifier/cascade.xml"

# load pretrained classifiers
knife_detector = cv2.CascadeClassifier(knife_detector_path)

frame = cv2.imread('../positives/knife.97.jpg')

#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# detect faces
items = knife_detector.detectMultiScale(frame,
                                        minNeighbors=5,
                                        minSize=(40, 40),
                                        flags=0,
                                        scaleFactor=1.5)

# Draw rectange around the faces
for (x, y, w, h) in items:

    print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
    print("length: {}, height: {}".format(w, h))
    print("")

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#frame = cv2.resize(frame, (960, 540))
#while True:
cv2.imshow('Video', frame)
#cv2.imwrite("item.jpg", frame)
# print(frame.shape)
# Exit condition ---> press esc to terminate video streaming
k = cv2.waitKey(0)
    # if k == 27:         # wait for ESC key to exit
    #     break
# Release resource and close image window
# cam.release()
cv2.destroyAllWindows()
