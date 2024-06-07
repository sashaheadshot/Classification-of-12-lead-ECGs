#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
from scipy.signal import resample


# In[2]:


with open ('saved_data.pkl','rb') as f:
    loaded_data = pickle.load(f)


# In[10]:


desired_sampling_rate = 500


resampled_data = {}

def get_sampling_rate(header_file):
    with open(header_file, 'r') as file:
        for line in file:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) > 2 and parts[2].isdigit():
                return int(parts[2])
    return None


for filename, data in loaded_data.items():
    if filename.endswith('.mat'):
        header_file = filename.replace('.mat', '.hea')
        
        sampling_rate = get_sampling_rate(header_file)
        
        if isinstance(data, dict) and 'val' in data:
            signal_data = data['val']
        
            num_points = int(signal_data.shape[1] * desired_sampling_rate / sampling_rate)
            resampled_signal = resample(signal_data, num_points, axis=1)
            resampled_data[filename] = {'val': resampled_signal}

# Now, resampled_data contains the signals resampled to the desired sampling rate of 500 Hz

for filename, data in resampled_data.items():
    signal = data['val']
    length_seconds = signal.shape[1] / desired_sampling_rate
    print(f"File: {filename}")
    print(f"Length of Signal: {length_seconds} seconds")
    print(f"Sampling Rate: {desired_sampling_rate} Hz\n")

# Save the processed signals back to .mat files if needed
for filename, data in resampled_data.items():
    savemat(filename, data)


# In[11]:


with open ('saved_resampled_data.pkl', 'wb')as f:
    pickle.dump(resampled_data,f)


# In[15]:


desired_length_seconds = 10

processed_data = {}

# Function to adjust signal length
def adjust_signal_length(signal, desired_length_samples):
    current_length_samples = signal.shape[1]
    if current_length_samples > desired_length_samples:
        # Trim the signal if it's longer than desired length
        processed_signal = signal[:, :desired_length_samples]
    else:
        # Pad the signal with zeros if it's shorter than desired length
        num_zeros_to_pad = desired_length_samples - current_length_samples
        processed_signal = np.pad(signal, ((0, 0), (0, num_zeros_to_pad)), mode='constant')
    return processed_signal


for filename, signal_data in resampled_data.items():
    desired_length_samples = desired_length_seconds * desired_sampling_rate
    processed_signal = adjust_signal_length(signal_data['val'], desired_length_samples)
    processed_data[filename] = {'val': processed_signal}


# In[16]:


for filename, data in processed_data.items():
    signal = data['val']
    length_seconds = signal.shape[1] / desired_sampling_rate
    print(f"File: {filename}")
    print(f"Length of Signal: {length_seconds} seconds")
    print(f"Sampling Rate: {desired_sampling_rate} Hz\n")


# In[17]:


with open ('saved_reshaped_data.pkl', 'wb')as f:
    pickle.dump(processed_data,f)

