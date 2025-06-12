import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_map = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}
colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)}  # green, red, orange


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("models/resnet18_mask_classifier.pth", map_location=device))
model.eval()
model.to(device)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


cap = cv2.VideoCapture(0)

print("🎥 Webcam başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = predicted.item()

        label_text = label_map[label]
        color = colors[label]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
