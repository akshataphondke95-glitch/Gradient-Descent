#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("G:/My Drive/Colab Notebooks/insurance.csv")


# In[3]:


df


# In[4]:


df_scaled=df.copy()
df_scaled['Age']=df_scaled['Age']/100
df_scaled


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df_scaled[['Age', 'Affordability' ]], df.Hava_insurance, test_size=0.2, random_state=2)


# In[6]:


len(y_train), len(y_test)


# In[7]:


model=keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)


# In[8]:


model.evaluate(X_test, y_test, 'epochs'==10)


# In[9]:


X_test


# In[10]:


model.predict(X_test)


# In[11]:


y_test


# In[12]:


coef, intercept=model.get_weights()
coef, intercept


# In[13]:


def sigmoid(x):
    import math
    return 1/(1+math.exp(-x))
sigmoid(18)


# In[14]:


def prediction_function(age, Affordability):
    weighted_sum=coef[0]*age+coef[1]*Affordability+intercept
    return sigmoid(weighted_sum)


# In[15]:


prediction_function(.54,1)


# In[16]:


prediction_function(.22,1)


# In[17]:


def log_loss(y_true, y_predicted):
    epsilon=1e-15
    y_predicted_new=[max(i, epsilon) for i in y_predicted]
    y_predicted_new=[max(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new=np.array( y_predicted_new)
    return -np.mean(y_true*np.log( y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


# In[18]:


def sigmoid_numpy(x):
    return 1/(1+np.exp(-x))

sigmoid_numpy(np.array([12, 0, 1]))


# In[19]:


def gradient_descent(Age, Affordability, y_true, epochs, loss_thresold=0.001):
    #w1 w2 bias
    w1=w2=1
    bias=0
    rate=0.5
    n=len(Age)
    for i in range(epochs):
        weighted_sum=w1*Age+w2*Affordability+bias
        y_predicted=sigmoid_numpy(weighted_sum)
        loss=log_loss(y_true, y_predicted)
        w1d=(1/n)*np.dot(np.transpose(Age),(y_predicted-y_true))
        w2d=(1/n)*np.dot(np.transpose(Affordability),(y_predicted-y_true))
        biass_d=np.mean(y_predicted-y_true)
        
        w1=w1-rate*w1d
        w2=w2-rate*w2d
        bias=bias-rate*biass_d
        
        print(f'Epoch:{i}, w1={w1}, w2:{w2}, bias:{bias}, loss:{loss}')
        
    return w1, w2, bias 


# In[20]:


gradient_descent(X_train['Age'], X_train['Affordability'], y_train, 10,loss_thresold=0.001)


# In[21]:


coef, intercept


# In[23]:


w1==0.9793, w2==0.9547, bias==-0.6463


# In[ ]:




