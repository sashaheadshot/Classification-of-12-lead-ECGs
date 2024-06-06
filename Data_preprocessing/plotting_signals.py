#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pickle


# In[2]:


with open ('saved_data.pkl','rb') as f:
    loaded_data = pickle.load(f)


# In[3]:


with open ('saved_filtered_data.pkl','rb') as f:
    filtered_data = pickle.load(f)


# In[4]:


# Unfiltered signal: 

mat_data = loaded_data['C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0004.mat']
ecg_signal = mat_data['val'].flatten()
ecg_signal_processed = ecg_signal[~np.isnan(ecg_signal)]
ecg_processed = nk.ecg_process(ecg_signal_processed)
lead_name = "ECG_Raw" 

plt.figure(figsize=(12, 4))
plt.plot(ecg_processed[0][lead_name])  
plt.title("Unfiltered ECG signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(1000, 5000)
plt.show()


# In[5]:


# Filtered Signal:

filtered_ecg_signal = filtered_data['C:\\Users\\Alex\\Downloads\\datasets\\training\\data\\cpsc_2018\\g1\\A0004.mat']

plt.figure(figsize=(12, 4))
plt.plot(filtered_ecg_signal)  
plt.title("Filtered ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(1000, 5000)
plt.show()

