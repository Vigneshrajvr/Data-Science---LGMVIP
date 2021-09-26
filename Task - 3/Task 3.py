#!/usr/bin/env python
# coding: utf-8
VIGNESHRAJ V

Data Science Intern-LGMVIP

Task 2 :Iris Flowers Classification ML Project 
    
Beginner Level Task


# In[1]:



import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:



iris = pd.read_csv(r'D:\vignesh\iris.data', header = None)


# In[4]:


iris


# In[5]:


print(iris.shape)


# In[7]:



print(iris.info)


# In[8]:


iris.isnull().sum


# In[9]:



iris.describe()


# In[10]:


iris0 = iris.rename(columns = { 0:'Sepal_Length'}, inplace = True)
iris1 = iris.rename(columns = { 1:'Sepal_Width'}, inplace = True)
iris2 = iris.rename(columns = { 2:'Petal_Length'}, inplace = True)
iris3 = iris.rename(columns = { 3:'Petal_Width'}, inplace = True)
iris4 = iris.rename(columns = { 4:'Species_Name'}, inplace = True)


# In[11]:


print(iris)


# In[24]:


rep = plt.figure()

axis_1 = rep.add_subplot(2,2,1)
axis_2 = rep.add_subplot(2,2,2)
axis_3 = rep.add_subplot(2,2,3)
axis_4 = rep.add_subplot(2,2,4)

axis_1.hist(iris['Sepal_Length'],color = 'green')
axis_1.set_xlabel('Sepal Length in cm')

axis_2.hist(iris['Sepal_Width'],color = 'red')
axis_2.set_xlabel('Sepal Width in cm')

axis_3.hist(iris['Petal_Length'],color = 'violet')
axis_3.set_xlabel('Petal Length in cm')

axis_4.hist(iris['Sepal_Length'],color = 'teal')
axis_4.set_xlabel('Petal Width in cm')

rep.set_figheight(12)
rep.set_figwidth(12)

plt.show()


# In[22]:


sns.pairplot(data = iris , kind = 'scatter')


# In[23]:


sns.pairplot(data = iris, hue = 'Species_Name', kind = 'reg')


# In[28]:


correlation = iris.corr()

plt.figure(figsize = (10,8))
sns.heatmap(correlation, annot = True, vmin = -1.0, cmap = 'viridis')
plt.title('Correlation Matrix of Iris Dataset')
plt.show()


# In[18]:


from sklearn.cluster import KMeans
list1 = []
a = iris[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width',]].to_numpy()

for i in range(1,11):
    kmeans = KMeans(n_clusters =i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    
    kmeans.fit(a)
    list1.append(kmeans.inertia_)


# In[19]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(a)


# In[31]:


plt.scatter(a[y_kmeans == 0, 0], a[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(a[y_kmeans == 1, 0], a[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(a[y_kmeans == 2, 0], a[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids')


plt.legend()


# In[ ]:




