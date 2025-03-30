import cv2
import numpy as np

# Load trained face recognizer model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.xml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define a dictionary of known people (Manually map IDs to names)
people = {1: "Navneet", 2: "Navneet"}

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_img)

        if confidence < 50:  # Lower confidence is better
            name = people.get(label, "Unknown")
            color = (0, 255, 0)  # Green for recognized faces
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized faces

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
