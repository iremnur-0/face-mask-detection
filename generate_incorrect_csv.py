import os
import pandas as pd


img_root = "archive/images/incorrect"
output_csv = "labels_extra.csv"
data = []


for subfolder in os.listdir(img_root):
    sub_path = os.path.join(img_root, subfolder)
    if os.path.isdir(sub_path):
        for filename in os.listdir(sub_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append({
                    "filename": filename,
                    "label": "mask_weared_incorrect",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 224,
                    "ymax": 224
                })

                os.system(f"cp '{os.path.join(sub_path, filename)}' archive/images/")


df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"{output_csv} oluşturuldu, {len(df)} adet incorrect etiket eklendi.")
