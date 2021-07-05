
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import math

from sknn.mlp import Classifier, Layer

DATA_FILE = 'DIGIT_RECOGNIZER/train.csv'
TEST_FILE = 'DIGIT_RECOGNIZER/test.csv'
SUBMISSION_FILE = 'DIGIT_RECOGNIZER/submit_CNN.csv'
df = pd.read_csv(DATA_FILE)
test_df = pd.read_csv(TEST_FILE)


# In[9]:

y_train, X_train = df['label'], df.ix[:,1:]
X_test = test_df


# In[10]:

'''
#Random forests
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
pd.DataFrame({"ImageId": range(1,len(preds)+1), "Label": preds}).to_csv(SUBMISSION_FILE, index=False, header=True)
'''


# In[20]:




# In[ ]:



