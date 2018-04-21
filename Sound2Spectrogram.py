# sound source를 spectrogram 으로 바꾸는파일입니다.
# 음원길이는 동일하게 30초로 구성되어있으며 1초씩 cut합니다.

import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import os

# load
y_list = []

#file directory
directory = os.listdir('data_rouen_sample')

#image cutting & melspectrogram
for file in directory:
    os.chdir('/Users/chan/Desktop/deeplearning/data_rouen_sample')
    fileName = str(os.path.basename(file))
    fileName = file.replace('.wav',"",1)
    y, sr = librosa.load(file)
    
    for k in range(30): #plt to img_file
        y_list = y[k*sr:(k+1)*sr] #y_vector의 sr개만큼이 1초를 구성        
        # melspectrogram
        S = librosa.feature.melspectrogram(y=y_list, sr=sr, n_mels=128, fmax=8000)
        
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        os.chdir('/Users/chan/Desktop/deeplearning/data_rouen_image')#directory for saving
        plt.savefig(fileName+"_"+str(k),bbox_inches='tight') #save

        
        
        
