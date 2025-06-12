import pandas as pd

df = pd.read_csv("labels.csv")
df = df[df["label"] != "label"]


with_mask = df[df["label"] == "with_mask"].sample(n=1000, random_state=42)
without_mask = df[df["label"] == "without_mask"].sample(n=1000, replace=True, random_state=42)  # 🔁 tekrar alınabilir
incorrect = df[df["label"] == "mask_weared_incorrect"].sample(n=1000, random_state=42)


balanced_df = pd.concat([with_mask, without_mask, incorrect], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


balanced_df.to_csv("labels_balanced.csv", index=False)
print("labels_balanced.csv güncellendi.")
print(balanced_df["label"].value_counts())
