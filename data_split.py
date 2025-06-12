import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split


df = pd.read_csv("labels.csv")


major_labels = df.groupby('filename')['label'].agg(lambda x: x.mode()[0]).reset_index()


train_val, test = train_test_split(major_labels, test_size=0.15, stratify=major_labels['label'], random_state=42)
train, val = train_test_split(train_val, test_size=0.176, stratify=train_val['label'], random_state=42)  # 0.176 x 0.85 ≈ 0.15

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)
os.makedirs("data/test", exist_ok=True)


def copy_images(df_subset, folder_name):
    for _, row in df_subset.iterrows():
        src = os.path.join("archive/images", row['filename'])
        dst = os.path.join("data", folder_name, row['filename'])
        shutil.copyfile(src, dst)

copy_images(train, "train")
copy_images(val, "val")
copy_images(test, "test")

print("Görseller başarıyla train/val/test olarak ayrıldı.")
