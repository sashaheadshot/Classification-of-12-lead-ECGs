#!/usr/bin/env python
# coding: utf-8

# In[7]:


import scipy.io
import os
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
import matplotlib.pyplot as plt


# In[2]:


main_directory = r'C:\Users\Alex\Downloads\datasets\training\data'
total_files = sum(len(files) for _, _, files in os.walk(main_directory))
progress_bar = tqdm(total = total_files, desc = "Data is loading...")
loaded_data = {}

for root, dirs, files in os.walk(main_directory):
    for filename in files:
        if filename.endswith('.mat') or filename.endswith('.hea'):
            file_path = os.path.join(root, filename)
            if filename.endswith('.mat'):
                data = scipy.io.loadmat(file_path)
            elif filename.endswith('.hea'):
                with open(file_path, 'r') as f:
                    data = f.read()
            loaded_data[file_path] = data
        progress_bar.update(1)
progress_bar.close()


# In[3]:


for i, key in enumerate(loaded_data.keys()):
    if i >= 100:
        break
    print(key)


# In[4]:


selected_key = 'C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0001.hea'

if selected_key in loaded_data:
    print(f"Contents of {selected_key}: \n")
    print(loaded_data[selected_key])
else: print(f"Selected key {selected_key} does not exist.")


# In[5]:


selected_key2 = 'C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0001.mat'
print(loaded_data[selected_key2])


# In[12]:


mat_data = loaded_data['C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0002.mat']
ecg_signal = mat_data['val'].flatten()
ecg_signal_processed = ecg_signal[~np.isnan(ecg_signal)]
ecg_processed = nk.ecg_process(ecg_signal_processed)
lead_name = "ECG_Raw" 

plt.figure(figsize=(12, 4))
plt.plot(ecg_processed[0][lead_name])  
plt.title("Processed ECG Signal (Lead I)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(1000, 5000)
plt.show()


# In[ ]:




