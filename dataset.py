import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceMaskDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        available_files = set(os.listdir(img_dir))
        self.annotations = self.annotations[self.annotations['filename'].isin(available_files)].reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")

        x_min = int(row['xmin'])
        y_min = int(row['ymin'])
        x_max = int(row['xmax'])
        y_max = int(row['ymax'])

        face = image.crop((x_min, y_min, x_max, y_max))

        label = row['label']
        label_map = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
        label = label_map[label]

        if self.transform:
            face = self.transform(face)

        return face, label
