import os
from nibabel import save, load
import pandas as pd
import nibabel
from pandas import read_csv
import numpy as np


data_dir = '/blue/stevenweisberg/ashishkumarsahoo/difumo/Data/freesurfer_121023/hippocampus_standardized'
save_dir = '/blue/stevenweisberg/ashishkumarsahoo/difumo/Data/freesurfer_121023/hippocampus_standardized_normalized'

subjects = read_csv(r'/blue/stevenweisberg/ashishkumarsahoo/hippocampAI/participants.tsv',sep='\t',header=0)
subjs = subjects.participant_id

        
for subj in subjs:
    subj_number = subj[4:]
    t1_image_MNI = load('{}/sub-{}_hippocampusmasked_brain_standardized.nii.gz'.format(data_dir,subj_number))
    isExist = os.path.exists('{}/{}'.format(data_dir,subj))
    if not isExist:
        os.makedirs('{}/{}'.format(data_dir,subj))
    img_data = t1_image_MNI.get_fdata()
    points = np.argwhere(img_data)
    # Normalize intensity with respect to the maximum value
    #max_value = np.max(img_data)
    min_value = np.min(img_data)
    print ('min value for', subj_number, ' :', min_value)
    max_value = np.max(img_data)
    print ('max value for', subj_number, ' :', max_value)
    #if min_value >= 0:
        #print('error')
    #print ('max value for', subj_number, ' :', max_value)
    #img_data = img_data-min_value #REMOVED SINCE NO NEGATIVE VALUES EXIST
    max_value = np.max(img_data)
    normalized_data = img_data / max_value
    if np.any(normalized_data < 0):
        print('error')
    np.save('{}/sub-{}_hippocampusmasked_brain_standardized_noneg_normalized.npy'.format(save_dir,subj_number), normalized_data)
    #print(subj)