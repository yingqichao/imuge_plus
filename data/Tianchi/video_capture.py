import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255))
        cv2.imshow("video", img)
        if cv2.waitKey(10) == ord('q'):
            break
    cv2.destroyAllWindows()
