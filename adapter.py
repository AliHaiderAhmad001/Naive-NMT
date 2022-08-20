# -*- coding: utf-8 -*-
"""adapter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F8IV1sVRla6Pr7-gTR4QtwKOc0SNqcW9
"""

import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/Naive_NMT')
from utils import preprocess_sentence,unicode_to_ascii
import pandas as pd

train_filenamepath ='/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/spa.txt'
INPUT_COLUMN = 'input'
TARGET_COLUMN = 'target'
NUM_SAMPLES = 80000
df=pd.read_csv(train_filenamepath, sep="\t", header=None, names=[INPUT_COLUMN,TARGET_COLUMN], usecols=[0,1], 
               nrows=NUM_SAMPLES)
input_data=df[INPUT_COLUMN].apply(lambda x : preprocess_sentence(x)).tolist()
target_data=df[TARGET_COLUMN].apply(lambda x : preprocess_sentence(x)+ ' <eos>').tolist()
target_input_data=df[TARGET_COLUMN].apply(lambda x : '<sos> '+ preprocess_sentence(x)).tolist()

# save
with open(r'/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/input_data.txt', 'w') as fp:
    fp.write('\n'.join(input_data))
with open(r'/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/target_data.txt', 'w') as fp:
    fp.write('\n'.join(target_data))
with open(r'/content/drive/MyDrive/Colab Notebooks/Naive_NMT/dataset/target_input_data.txt', 'w') as fp:
    fp.write('\n'.join(target_input_data))