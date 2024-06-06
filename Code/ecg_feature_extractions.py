#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import pickle
from tqdm import tqdm


# In[5]:


with open ('saved_filtered_data.pkl','rb') as f:
    filtered_data = pickle.load(f)


# In[32]:


# R peak indecies:

select_key = 'C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0001.mat'

ecg_signal = filtered_data[select_key].flatten()
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=500)


print("R-peaks locations:", rpeaks)


# In[33]:


# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)


# In[42]:


select_key = 'C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0002.mat'
filtered_ecg_signal = filtered_data[select_key]
sampling_rate = 500

_, filtered_rpeaks = nk.ecg_peaks(filtered_ecg_signal, sampling_rate=sampling_rate)


plt.figure(figsize=(15, 6)) 
plot = nk.events_plot(filtered_rpeaks['ECG_R_Peaks'], filtered_ecg_signal)
plt.xlim(0, len(filtered_ecg_signal) / 30)
plt.show()


# In[43]:


# Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(filtered_ecg_signal, filtered_rpeaks, sampling_rate=sampling_rate, method="peak")

# Zooming into the first 3 R-peaks, with focus on T-peaks, P-peaks, Q-peaks, and S-peaks
plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3], 
                       waves_peak['ECG_P_Peaks'][:3],
                       waves_peak['ECG_Q_Peaks'][:3],
                       waves_peak['ECG_S_Peaks'][:3]], filtered_ecg_signal[:500])
plt.show()


# In[ ]:


#Classes: 
    #'426783006': 'Sinus Rhythm',
    #'427084000': 'Sinus Tachycardia',
    #'426177001': 'Sinus Bradycardia',
    #'427172004': 'PVC',
    #'270492004': '1st Degree AV block',
    #'164865005': 'Myocardial infarction',
    #'55930002': 'ST changes'


# In[44]:


ecg_features = {}
progress_bar = tqdm(total=len(filtered_data), desc="Extracting Features")

# Function to safely convert to list and handle NumPy types
def safe_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.float64)):  # Convert NumPy integers and floats to Python types
        return obj.item()
    elif isinstance(obj, list):
        return [safe_to_list(i) for i in obj]  # Recursively handle lists
    else:
        return obj


for filepath, filtered_signal in filtered_data.items():
    try:
        _, filtered_rpeaks = nk.ecg_peaks(filtered_signal, sampling_rate=sampling_rate)
        _, waves_peak = nk.ecg_delineate(filtered_signal, filtered_rpeaks, sampling_rate=sampling_rate, method="peak")
        ecg_features[filepath] = {
            "ECG_R_Peaks": safe_to_list(filtered_rpeaks['ECG_R_Peaks']),
            "ECG_T_Peaks": safe_to_list(waves_peak['ECG_T_Peaks']),
            "ECG_P_Peaks": safe_to_list(waves_peak['ECG_P_Peaks']),
            "ECG_Q_Peaks": safe_to_list(waves_peak['ECG_Q_Peaks']),
            "ECG_S_Peaks": safe_to_list(waves_peak['ECG_S_Peaks'])
        }
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

    progress_bar.update(1)

progress_bar.close()


output_json_path = 'C:\\Users\\Alex\\Downloads\\datasets\\Extra\\test_1.json'
with open(output_json_path, 'w') as json_file:
    json.dump(ecg_features, json_file)

print(f"ECG features saved to {output_json_path}")


# In[ ]:




