import io
import re
import csv
import math
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

df = pd.read_csv("out_scaled_400.csv", header=None)
print("term document shape:", df.shape)

def cosine_similarity(df, query, tol):
    similars = []
    for i in range(df.shape[1]):
        a = np.array(df.iloc[:, i])
        query = np.array(query)
        similarity_val = np.dot(a, query) / (np.linalg.norm(query) * np.linalg.norm(a))
        if similarity_val > tol:
            similars.append(i)
    return similars


def get_original_labels(path):
    labels = []
    for i in range(31):
        labels.append([])
    with open(path) as f:
        temp = f.read().splitlines()
        for t in temp:
            x = t.split()
            labels[int(x[0]) - 1].append(int(x[2]) - 1)
    return labels

def calculate(df, query_number):
    labels = get_original_labels('MED_REL_1')
    train = df.iloc[:,:399]
    precisions = []
    recalls = []
    for i in np.linspace(1, 0, 26):
        q = df.iloc[:,int(query_number) + 398]
        similar_to_i = cosine_similarity(train, q, i)
        print("i = ", i, "found = ", len(similar_to_i))
        y_preds = [0] * 1033
        y_test = [0] * 1033
        for test in similar_to_i:
            y_preds[test] = 1
        for test in labels[query_number - 1]:
            y_test[test] = 1
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_preds)
        precisions.append(precision[1])
        recalls.append(recall[1])
        
    return precisions, recalls


precisions, recalls = calculate(df, 1)
precision_original = [1.0 if x == 0.0 else x for x in precisions]
recalls_original = recalls

s = pd.read_csv('S.csv', header=None)
u = pd.read_csv('U.csv', header=None)
v = pd.read_csv('V.csv', header=None)
u = u.iloc[:,:-1]
v = v.iloc[:,:-1]
u = u.values
v = v.values
s = s[0]

prec_total = []
rec_total = []
for i in range(100, 401, 100):
    a = u[:, :i]
    print(a)
    b = np.diag(s[:i])
    print(b)
    c = v[:i, :]
    temp = a @ b @ c
    temp = pd.DataFrame(temp, index=None)
    print(temp.shape)
    prec, rec = calculate(temp, 1)
    print("i = ", i)
    print("prec = ", prec)
    prec_new = [1.0 if x == 0.0 else x for x in prec]
    print("recall = ", rec)
    prec_total.append(prec_new)
    rec_total.append(rec)

ranks = [100, 200, 300, 400]
for i in range(len(prec_total)):
	fig= plt.figure(figsize=(10,10))
	plt.title("Precision vs Recall (changing tolerance from 1 to 0)")
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.plot(recalls_original, precision_original, label='original', linewidth = 2)
	plt.plot(rec_total[i], prec_total[i], label=str(ranks[i]) + 'rank', linestyle='dashed', linewidth = 2)
	plt.legend()
	plt.show()

