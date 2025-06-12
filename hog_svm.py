import os
import cv2
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog


df = pd.read_csv("labels.csv")
df = df[df['filename'].apply(lambda f: os.path.exists(os.path.join("data/train", f)))]

features = []
labels = []

label_map = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}

for _, row in df.iterrows():
    path = os.path.join("data/train", row["filename"])
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = gray[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
    face = cv2.resize(face, (64, 64))

    hog_feat = hog(face, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)

    features.append(hog_feat)
    labels.append(label_map[row["label"]])


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = LinearSVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["with_mask", "without_mask", "incorrect"]))
