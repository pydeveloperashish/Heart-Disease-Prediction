#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn
import numpy as np
import pandas as pd
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go

import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot


# In[3]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[4]:


heart=pd.read_csv(r'C:\Python37\Projects\ALL ML-DL-DS Projects from Udemy and other Sources\Data_Science\ML_Casestudies-master\heart disease\heart.csv')


# In[5]:


heart


# In[6]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(heart.columns[i]+":\t\t\t"+info[i])


# In[ ]:





# In[7]:


heart['target']


# In[8]:


heart.groupby('target').size()


# In[9]:


heart.groupby('target').sum()


# In[10]:


heart.shape


# In[11]:


heart.size


# In[12]:


heart.describe()


# In[ ]:





# In[13]:


heart.info()


# In[14]:


heart['target'].unique()


# In[ ]:





# In[15]:


#Visualization


# In[16]:


heart.hist(figsize=(14,14))
plt.show()


# In[ ]:





# In[17]:


plt.bar(x=heart['sex'],height=heart['age'])
plt.show()


# In[18]:


sns.barplot(x="fbs", y="target", data=heart)
plt.show()


# In[19]:


sns.barplot(x=heart['sex'],y=heart['age'],hue=heart['target'])


# In[20]:


sns.barplot(heart["cp"],heart['target'])


# In[21]:


sns.barplot(heart["sex"],heart['target'])


# In[ ]:





# In[25]:


px.bar(heart,heart['sex'],heart['target'])


# In[26]:


sns.distplot(heart["thal"])


# In[27]:


sns.distplot(heart["chol"])


# In[28]:


sns.pairplot(heart,hue='target')


# In[29]:


heart


# In[30]:


numeric_columns=['trestbps','chol','thalach','age','oldpeak']


# In[31]:


sns.pairplot(heart[numeric_columns])


# In[32]:


heart['target']


# In[33]:


y = heart["target"]

sns.countplot(y)

target_temp = heart.target.value_counts()

print(target_temp)


# In[ ]:





# In[34]:


# create a correlation heatmap
sns.heatmap(heart[numeric_columns].corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[ ]:





# In[ ]:





# In[35]:


# create four distplots
plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(heart[heart['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(heart[heart['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(heart[heart['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(heart[heart['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()


# In[36]:


plt.figure(figsize=(13,6))
plt.subplot(121)
sns.violinplot(x="target", y="thalach", data=heart, inner=None)
sns.swarmplot(x="target", y="thalach", data=heart, color='w', alpha=0.5)


plt.subplot(122)
sns.swarmplot(x="target", y="thalach", data=heart)
plt.show()


# In[ ]:





# In[37]:


heart


# In[38]:


# create pairplot and two barplots
plt.figure(figsize=(16,6))
plt.subplot(131)
sns.pointplot(x="sex", y="target", hue='cp', data=heart)
plt.legend(['male = 1', 'female = 0'])
plt.subplot(132)
sns.barplot(x="exang", y="target", data=heart)
plt.legend(['yes = 1', 'no = 0'])
plt.subplot(133)
sns.countplot(x="slope", hue='target', data=heart)
plt.show()


# In[ ]:





# In[ ]:





# In[39]:


#DATA Preprocessing


# In[40]:


########################################################################################


# In[41]:


heart['target'].value_counts()


# In[42]:


heart['target'].isnull()


# In[43]:


heart['target'].sum()


# In[44]:


heart['target'].unique()


# In[45]:


####################################################################################################3


# In[ ]:





# In[46]:


heart.isnull().sum()


# In[ ]:





# In[ ]:





# In[47]:


#Storing in X and y


# In[48]:


X,y=heart.loc[:,:'thal'],heart.loc[:,'target']


# In[49]:


X


# In[50]:


y


# In[51]:


####Or X, y = heart.iloc[:, :-1], heart.iloc[:, -1]


# In[52]:


X.shape


# In[53]:


y.shape


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[55]:


X=heart.drop(['target'],axis=1)


# In[56]:


#X=np.array(X)


# In[57]:


X


# In[58]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)


# In[59]:


X_test


# In[60]:


y_test


# In[ ]:





# In[ ]:





# In[61]:


print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(y_test.shape))


# In[ ]:





# In[ ]:





# In[ ]:





# In[62]:


#Model


# In[63]:


#Decision Tree Classifier


# In[64]:


Catagory=['No....but i pray you get Heart Disease or at leaset Corona Virus Soon...','Yes you have Heart Disease....RIP in Advance']


# In[65]:


from sklearn.tree import DecisionTreeClassifier


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[66]:


prediction=dt.predict(X_test)
accuracy_dt=accuracy_score(y_test,prediction)*100


# In[67]:


accuracy_dt


# In[68]:


print("Accuracy on training set: {:.3f}".format(dt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))


# In[ ]:





# In[69]:


y_test


# In[70]:


prediction


# In[ ]:





# In[71]:


X_DT=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
X_DT_prediction=dt.predict(X_DT)


# In[72]:


X_DT_prediction[0]


# In[73]:


print(Catagory[int(X_DT_prediction[0])])


# In[ ]:





# In[74]:


#Feature Importance in Decision Trees


# In[75]:


print("Feature importances:\n{}".format(dt.feature_importances_))


# In[76]:


def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_diabetes(dt)
plt.savefig('feature_importance')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:


# KNN


# In[78]:


sc=StandardScaler().fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# In[79]:


X_test_std


# In[80]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_std,y_train)


# In[81]:


prediction_knn=knn.predict(X_test_std)
accuracy_knn=accuracy_score(y_test,prediction_knn)*100


# In[82]:


accuracy_knn


# In[83]:


print("Accuracy on training set: {:.3f}".format(knn.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))


# In[ ]:





# In[84]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[85]:


scores


# In[86]:


plt.plot(k_range,scores_list)


# In[87]:


px.line(x=k_range,y=scores_list)


# In[ ]:





# In[ ]:





# In[88]:


X_knn=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
X_knn_std=sc.transform(X_knn)
X_knn_prediction=dt.predict(X_knn)


# In[89]:


X_knn_std


# In[90]:


(X_knn_prediction[0])


# In[91]:


print(Catagory[int(X_knn_prediction[0])])


# In[ ]:





# In[92]:


algorithms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]


# In[93]:


sns.set(rc={'figure.figsize':(15,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[ ]:





# In[ ]:





# In[ ]:




