
# coding: utf-8

# In[134]:


import sklearn
import json
import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import scipy.stats as stats
import cPickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# In[157]:


IP_train = open('training.json', 'rb')
train = []
subjects = ["English", "Physics", "Chemistry", "ComputerScience", "Biology", "PhysicalEducation", "Economics", "Accountancy", "BusinessStudies", "Mathematics"]

for lines in IP_train.readlines():
    try:
        lines = json.loads(lines.strip("\n"))
        del lines['serial']
        marks = []
        for sub in subjects:
            try:
                marks.append(lines[sub])
            except:
                continue
        if len(marks) < 5:
            continue
        train.append(marks)
    except:
        pass


# In[158]:


train = pd.DataFrame(train)
train.columns = ["Sbj1", "Sbj2", "Sbj3", "Sbj4", "Sbj5"]


# In[160]:


train = pd.DataFrame(train)
train.columns = ["Sbj1", "Sbj2", "Sbj3", "Sbj4", "Sbj5"]
train = train.fillna(train.mean())
train["Sbj4"] = train["Sbj4"].astype(int)
train["Sbj5"] = train["Sbj5"].astype(int)

X_train, y_train = train.ix[:,:3], train["Sbj5"]
lreg_sci = LinearRegression(normalize=True)
lreg_sci.fit(X_train, y_train)


# In[172]:


import json
n = int(raw_input())
test = []

subjects = ["English", "Physics", "Chemistry", "ComputerScience", "Biology", "PhysicalEducation", "Economics", "Accountancy", "BusinessStudies"]
for i in xrange(0, n):
    text = json.loads(raw_input())
    del text['serial']
    marks = []
    for sub in subjects:
        try:
            marks.append(text[sub])
        except:
            continue
        if len(marks) < 4:
            continue
        test.append(marks)
        
test = pd.DataFrame(test)
test.columns = ["Sbj1", "Sbj2", "Sbj3", "Sbj4"]
test = test.fillna(test.mean())
X_test = test.ix[:,:3]
pred = lreg_sci.predict(X_test)

for i in pred:
    print int(i)


# In[ ]:




