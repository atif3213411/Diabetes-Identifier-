#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[3]:


dataset = pd.read_csv('../Downloads/diabetes.csv')


# In[4]:


len(dataset)
dataset.head()


# In[5]:


#Now we need to remove some of the useless data 
zero_useless = ['Glucose' , 'BloodPressure' , 'SkinThickness' , 'BMI' , 'Insulin']


# In[6]:


for column in zero_useless:
    dataset[column] = dataset[column].replace(0 , np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN , mean)


# In[8]:


#split the data set 
X = dataset.iloc[: , 0:8]
y = dataset.iloc[: , 8]
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 0 , test_size = 0.2)


# In[9]:


print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))


# In[11]:


##Feature Scaling 
sc_x = StandardScaler()


# In[12]:


X_train = sc_x.fit_transform(X_train)


# In[13]:


X_test = sc_x.fit_transform(X_test)


# In[14]:


##Now splitting dataset -> Scaling dataset is done
##Now we will train model 


# In[15]:


classifier = KNeighborsClassifier(n_neighbors=11, p=2,metric='euclidean')


# In[16]:


classifier.fit(X_train , y_train)


# In[17]:


##Model is trained 
#Now we need to predict the results 


# In[18]:


y_pred = classifier.predict(X_test)


# In[19]:


y_pred


# In[20]:


##Now to analyse this prediction  , we will make a confusion matrix 


# In[21]:


cm = confusion_matrix(y_test , y_pred)


# In[22]:


print(cm)


# In[23]:


##Now we want to find the f1-score 
x = f1_score(y_test , y_pred)


# In[24]:


print(x)


# In[ ]:




