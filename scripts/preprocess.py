import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. This automatically finds the folder where THIS script is saved
# and then goes to the 'data' folder relative to it.
script_dir = os.path.dirname(os.path.abspath(__file__)) # location of 'scripts'
project_root = os.path.dirname(script_dir)             # location of 'HPC-intrusion'
file_path = os.path.join(project_root, 'data', 'Tuesday-WorkingHours.pcap_ISCX.csv')
output_path = os.path.join(project_root, 'data', 'cleaned_tuesday.csv')

print(f"Targeting file at: {file_path}")

# 2. Load the raw data
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("File found and loading...")
else:
    print("ERROR: Still can't find the file. Let's check the directory contents:")
    print(os.listdir(os.path.join(project_root, 'data')))
    exit()

# 3. Clean column names (strip spaces)
df.columns = df.columns.str.strip()

# 4. Handle Infinity and NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 5. Encode the Labels
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# 6. Scale features (this part might take a minute because the file is 135MB)
scaler = MinMaxScaler()
features = df.drop('Label', axis=1)
df[features.columns] = scaler.fit_transform(features)

# 7. Save the "Clean" version (header=False is better for C++)
df.to_csv(output_path, index=False, header=False)
print(f"Success! Cleaned data saved to: {output_path}")