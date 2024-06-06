#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
from tqdm import tqdm


# In[34]:


labels_dict = {}

for root, dirs, files in os.walk(main_directory):
    for filename in files:
        if filename.endswith('.hea'):
            try:
                with open(os.path.join(root, filename), 'r') as f:
                    header_data = f.readlines()
                
                disease_numbers = []
                for line in header_data:
                    if line.startswith('# Dx'):
                        disease_numbers = line.split(': ')[1].strip().split(',')  # Assuming multiple disease numbers separated by comma
                        break

                # Map the disease numbers to class labels
                labels = [class_mapping[disease_number.strip()] for disease_number in disease_numbers if disease_number.strip() in class_mapping]
                
                if labels:
                    labels_dict[filename.replace('.hea', '')] = labels
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

output_csv_path = 'C:\\Users\\Alex\\Downloads\\datasets\\labels_multiple_classes.csv'

with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Labels'])  # Write the header
    for filename, labels in labels_dict.items():
        writer.writerow([filename, ', '.join(labels)])

print(f"Labels saved to {output_csv_path}")


# In[ ]:




