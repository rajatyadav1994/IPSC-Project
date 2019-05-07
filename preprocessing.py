import io
import re
import csv
import math
import numpy as np
import pandas as pd
import copy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

fp = open("MED_400", "r")
itr = -1
start = 0
data = []
for line in fp.readlines():
    if ".I" in line:
        itr += 1
    elif ".W" in line:
        start = 1
        continue
    elif start == 1:
        start = 0
        data.append(line)
        data[itr] += ' '
    else:
        data[itr] += line
        data[itr] += ' '
fp2 = open("MED_400_q", "r")
for line in fp2.readlines():
    if ".I" in line:
        itr += 1
    elif ".W" in line:
        start = 1
        continue
    elif start == 1:
        start = 0
        data.append(line)
        data[itr] += ' '
    else:
        data[itr] += line
        data[itr] += ' '

print("Data Length = ", len(data))

ps = PorterStemmer()
with open("stopwords.txt") as f:
    temp = f.read().splitlines() 

stop_words = [ps.stem(w) for w in temp]

#stop_words = set(stopwords.words('english'))
data_tokens = []
ps = PorterStemmer()
ls = LancasterStemmer()
for dat in data:
#     dat = re.sub("\.", "", dat)
#     dat = re.sub("\,", "", dat)
#     dat = re.sub("[\,]", "", dat)
    dat = re.sub(r"([\.\,\-\(\)\\*\'\"])", "", dat)
    dat = re.sub(r"[\=\\\/]", "", dat)
    dat = re.sub(r"[0-9]+", "", dat)
    word_tokens = word_tokenize(dat)
    filtered = [ps.stem(w) for w in word_tokens if not ps.stem(w) in stop_words]
    filtered = [ps.stem(w) for w in filtered if not w in stop_words]
    data_tokens.append(filtered)

words = []
for doc in data_tokens:
    for word in doc:
        if word not in words:
            words.append(word)

print("Total Terms = ", len(words))


#Term Document Matrix
tdm = []
for doc in data_tokens:
    doc_counts = [0.0] * len(words)
    for word in doc:
        doc_counts[words.index(word)] += 1
    tdm.append(doc_counts)

tdm = np.array(tdm)
tdm = tdm.T

with open("out_400.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(tdm)

new = copy.deepcopy(tdm)
for i in range (new.shape[0]):
    for j in range (new.shape[1]):
        new[i][j] = new[i][j] * math.log10(new.shape[1] /  np.count_nonzero(new[i]))

print("Term Document Matrix shape:", new.shape)

with open("out_scaled_400.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(new)

