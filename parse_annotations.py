import os
import xml.etree.ElementTree as ET
import pandas as pd

ANNOTATIONS_DIR = 'archive/annotations'
IMAGES_DIR = 'archive/images'


data = []

for file in os.listdir(ANNOTATIONS_DIR):
    if file.endswith('.xml'):
        path = os.path.join(ANNOTATIONS_DIR, file)
        tree = ET.parse(path)
        root = tree.getroot()

        filename = root.find('filename').text


        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            data.append({
                'filename': filename,
                'label': label,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })


df = pd.DataFrame(data)


df.to_csv('labels.csv', index=False)
print("labels.csv generated.")
