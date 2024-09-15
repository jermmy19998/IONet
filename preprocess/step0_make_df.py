import os
import pandas as pd
from sklearn.model_selection import train_test_split

paths_labels = [
    # replace path and label to your own
    ('your path to dataset','lablename (e.g. HGSC or other)')
]


df = pd.DataFrame(columns=['image_id', 'label', 'image_path'])


for path, label in paths_labels:
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith('.svs'):
                file_path = os.path.join(path, filename)
                image_id = filename.split('.')[0]
                df = pd.concat([df, pd.DataFrame({'image_id': [image_id], 'label': [label], 'image_path': [file_path]})], ignore_index=True)
    else:
        print(f"Path does not exist: {path}")





df.drop_duplicates()
print(df)
df.to_csv('./out_csv/OC_lc.csv', index=False)



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('./out_csv/lc_train_40x.csv', index=False)
test_df.to_csv('./out_csv/lc_test_40x.csv', index=False)

print("Training DataFrame shape:", train_df.shape)
print("Testing DataFrame shape:", test_df.shape)
