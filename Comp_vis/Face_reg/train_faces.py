import cv2
import os
import numpy as np

# Create a directory to store training images
if not os.path.exists("dataset"):
    os.makedirs("dataset")

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

person_id = input("Enter a numeric ID for this person: ")
num_samples = 0
max_samples = 50  # Capture 50 images per person

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        num_samples += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/user_{person_id}_{num_samples}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q") or num_samples >= max_samples:
        break

cam.release()
cv2.destroyAllWindows()

print("Face images collected. Now training the model...")

# Train the model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels = [], []

for filename in os.listdir("dataset"):
    if filename.endswith(".jpg"):
        label = int(filename.split("_")[1])  # Extract user ID from filename
        img_path = os.path.join("dataset", filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label)

faces = np.array(faces, dtype="object")
labels = np.array(labels)

face_recognizer.train(faces, labels)
face_recognizer.save("face_model.xml")

print("Training complete! Model saved as 'face_model.xml'.")
