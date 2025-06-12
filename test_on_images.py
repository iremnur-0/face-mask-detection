import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}


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


test_folder = "test_images"


for img_file in os.listdir(test_folder):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_folder, img_file)
        image = Image.open(img_path).convert("RGB")


        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        right = left + side
        bottom = top + side
        image = image.crop((left, top, right, bottom))

 
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            label = label_map[pred.item()]


        plt.imshow(image)
        plt.title(f"Prediction: {label}")
        plt.axis('off')
        plt.show()

