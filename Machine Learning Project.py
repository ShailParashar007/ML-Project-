#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project

# In[1]:


#First step to read the file and do EDA(Exploratory Data Analysis) and Pre-Processing


# In[2]:


import pandas as pd                      # Importing libraries, pandas for file reading,                                        
import matplotlib.pyplot as plt          # Matplotlib and seaborn for visualisation
import seaborn as sns
import numpy as np                       #Numpy for array operations


# In[3]:


data=pd.read_csv('C://Users//shail//Desktop//Study Folder//Python Code//ML Project//Freelance ML Project.csv') 
#reading the file and displaying file
data.head()


# # After reading any data we do following operation to process the data
# 1-Pulling out basic info of dataset like how many columns are there of which data types, Basically we look at the pattern of data set
#    
# 2-Elimination or Replacing of Null Values
# 
# 3-Removal of Skewness with help of Boxcox Transformation & Scaling of data
# 
# 4-Elimination of Outliers
# 
# 5-Categorical Data Encoding-One Hot Encoding or Label Encoding
# 
# 6-Data Normalization and Scaling

# # (1) Basic EDA Process

# In[4]:


data.info()  # This gives info of columns ,Their total counts and data types.


# In[5]:


data.shape
#Number of rows=12222
#number of coloumn=17


# In[6]:


data.describe()   # describes the data of each column having numeric values


# In[7]:


data.drop('Sub Category Name',axis=1,inplace=True)  # Removing those columns which are not so important in predicting 'Type' 
data.drop('Description',axis=1,inplace=True)       # as well as those column are having high variance which can affect the model
data.drop('Client Registration Date',axis=1,inplace=True)


# # (2) Handling Null Values
# 
# #### Null values causes trouble in ML algorithm performance

# In[8]:


data.isnull().sum()  # Finding how many null values are there


# Here Duration and Client Job Title is having null values.We can either delete the null values or we can replace them with mean or any other value (Imputation). In this situation we 
# cannot delete the values as quantity is very high so we will go for imputation. We'll replace Null values with 'NA' in Duration as well as in  Client Job title

# In[9]:


null_index=data.index[data['Duration'].isnull()]  #We will take index of rows having null values
null_index


# In[10]:


# Replacing null values with 'NA'
data['Duration'].fillna('NA', inplace=True)
data['Client Job Title'].fillna('NA', inplace=True)


# In[11]:


data.iloc[null_index]  # As we can see the null places are filled with 'NA'


# In[12]:


data.dtypes       # To check the data types in each coloumn


# In[13]:


#Now checking for null values in whole data
data.isnull().sum()


# All the Null Values are eliminated

# # (3) Handling Skewness
# 

# In[14]:


plt.hist(x=data['Budget'],bins=10,edgecolor='black') # Removing skewness in order to make regression model on budget
plt.show()


# In[15]:


plt.figure(figsize=(5,5))             # As we can see the data isn't normally distributed
sns.distplot(data['Budget'],bins=40).set(title='Distribution of Budget')
plt.show()


# In[16]:


sns.boxplot(data=data,y='Budget')   #Black points reflects outliers


# In[17]:


data['Budget'].skew()  #Positive-side skweness


# In[18]:


data['Budget'].shape


# In[19]:


from sklearn.preprocessing import MinMaxScaler  # Importing MinMax package to scale the data. 
                                                # Scaling of data removes ouliers to some extent

scaler=MinMaxScaler()
data['Budget']=scaler.fit_transform(data['Budget'].values.reshape(-1, 1))
data['Budget']


# In[20]:


from scipy.stats import boxcox # Importing Boxcox, Boxcox also helps in making the data to normal distribution.
budget_transformed,lambda_value=boxcox(data['Budget']+0.00000001)  # Removing zero value as Boxcox doesn't accept zero value.
budget_transformed


# In[21]:


# visualising distribution before and after Box-cox Transformation

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
plt.hist(data['Budget'],bins=10,edgecolor='black',color='Red')
plt.title('Histogram of Budget')

plt.subplot(2,2,2)
sns.distplot(data['Budget'],color='Red')
plt.title('Distribution of Budget')

plt.subplot(2,2,3)
plt.hist(budget_transformed,bins=20,edgecolor='black')
plt.title('Histogram of Budget after Box-Cox')

plt.subplot(2,2,4)
sns.distplot(budget_transformed)
plt.title('Distribution of Budget after Box-Cox')
plt.show()


# In[22]:


data['Budget']=budget_transformed  #Putting Box-Cox Values in Budget


# In[23]:


data['Budget'].skew()  # skewness is decreased


# # (3) Handling Outliers

#  Outliers are those values which are either very large or very small from values next to it.  We can find outliers with the help of Boxplot. By putting condition (<>) we can remove them.
# 
# 

# In[24]:


plt.boxplot(data['Budget'])
plt.show()

data['Budget'].gt(-4).value_counts() # as we can see that count of data greater than -4 is very less (239) 
                                     # so we can remove them.
    
outliers_index=data.index[(data['Budget']>-4)|(data['Budget']<-8)]  #Taking index of those points which are >-4 and <-8
outliers_index 

data.drop(outliers_index,axis=0,inplace=True) #deleting the ouliers

plt.boxplot(data['Budget'])  # Outliers are removed
plt.show()


# In[25]:


data['Budget'].skew()  # By removing ouliers skewness is also decreased


# In[26]:


col_obj=data.select_dtypes('object').columns   # taking those columns which are of object data type
col_obj


# In[27]:


from sklearn.preprocessing import LabelEncoder   #Using LabelEncoder to transform the 


for x in col_obj:
    lb=LabelEncoder()
    data[x]=lb.fit_transform(data[x])


# In[28]:


newdata=data  #Saving the copy of data to make classification model


# In[29]:


data.dtypes # To know data type of each column


# # Spliting the data into training and testing

# In[30]:


from sklearn.model_selection import train_test_split
xtrain,xtest=train_test_split(data,train_size=0.8,random_state=4)  #training data is 80% of whole data


# In[31]:


xtrain.shape  # rows=9460, column=13


# In[32]:


xtest.shape   # rows=2365, column=13


# # 1. Making Clusters of Projects which are of  similar types

#  To group the similar project we will use K-Means Clustering algorithm 

#  As you can see we are having 13 columns which can create curse of dimensionality so with the help of  Principal Component Analysis we will reduce the dimension to 2

# ## Principal Component Analysis

# In[33]:


from sklearn.decomposition import PCA

pca=PCA(n_components=2) #we have put 2 because we want data in 2D format (We can visualize easily in 2D)
xtrain=pca.fit_transform(xtrain) #perform pca on training set
xtest=pca.transform(xtest) #perform pca on testing set

print(xtrain.shape)
print(xtest.shape)


# #### Here you can see columns are reduced from 13 to 2

# In[34]:


sum(pca.explained_variance_ratio_) # 98% of variance is preserved while transforming


# ## Using Elbow method to find best no. of Centroids

# To find the best number of centroids we will plot Elbow Graph

# In[35]:


from sklearn.cluster import KMeans #import clustering module from sklearn
WCSSlist = []
for i in range (1,11):    #creating loop to find best value
    model = KMeans(n_clusters = i)
    model.fit(xtrain)
    WCSS = model.inertia_ #calculate Within-Cluster-Sum-of-Squares
    WCSSlist.append(WCSS)
print(WCSSlist)    


# In[36]:


# Plotting Elbow Graph 

plt.title('Elbow Graph')
plt.plot(range(1,11),WCSSlist)
plt.xlabel('Number of Centroids')
plt.ylabel('WCSS Value')
plt.show()


# We can clearly see that after 4 centroids WCSS is not reducing gradually, Therefore we are going to make centroids

# ### Creating model with 4 clusters

# In[37]:


model=KMeans(n_clusters=4)
model.fit(xtrain)
pred=model.predict(xtest)
pred[:5]


# In[38]:


print(pd.Series(pred).value_counts())  # Counts in each segment


# In[39]:


model.cluster_centers_  # Co-ordinates of centroids


# In[40]:


# Creating visualization for Clusters

plt.scatter(xtest[pred==0, 0],xtest[pred==0, 1],s=20,c='cyan',label='Cluster 1')
plt.scatter(xtest[pred==1, 0],xtest[pred==1, 1],s=20,c='green',label='Cluster 2')
plt.scatter(xtest[pred==2, 0],xtest[pred==2, 1],s=20,c='magenta',label='Cluster 3')
plt.scatter(xtest[pred==3, 0],xtest[pred==3, 1],s=20,c='brown',label='Cluster 4')

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=80,c='Black',label='Centroids')


plt.title('Cluster of Similar Projects',weight='bold')
plt.xlabel('Feature 1', weight='heavy')
plt.ylabel('Feature 2', weight='heavy')
plt.legend()
plt.show()


# In[41]:


x=data.drop(columns=['Budget'])   #Budget is our taget column 
y=data['Budget']


# In[42]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=4)


# Before going forward with RamdomForest , Logistic regression, Ridge and Lasso was applied but R2 Score as not more than 0.62
#  
# Reason why RandomForest gives best result because it works on the concept of bagging. In bagging, a group of models is trained on different subsets of the dataset, and the final output is generated by collating the outputs of all the different models.
# 
# This data set is very large having 11825 rows and 13 columns 
#  

# In[43]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor()
model.fit(xtrain, ytrain) 
trainpred = model.predict(xtrain) 
testpred = model.predict(xtest)



print('R2 Score of Training  :', np.round(r2_score(ytrain,trainpred),2))
print('R2 Score of Testing :', np.round(r2_score(ytest,testpred),2))
print()


print('Mean Squared Error of Training  :', np.round(mean_absolute_error(ytrain,trainpred),2))
print('Mean Squared Error of Testing :', np.round(mean_absolute_error(ytest,testpred),2))


# In[44]:


from yellowbrick.regressor import PredictionError  #Importing Yellowbrick package to visualize Error
model=RandomForestRegressor()
visualizer=PredictionError(model)
visualizer.fit(xtrain,ytrain)
visualizer.score(xtest,testpred)
visualizer.show()


# Yellowbrick helps in visualising the error we got between actual & prediction

# # 3.Classification Model to Predict 'Type'

# Here target column is to predict the 'Type' of the job ie.fixed price or hourly

# Using Random Forest Classifier as it is best model. It takes subsets of data and train multipe desicion trees. It has a ability to reduce the variance without increasing the bias

# In[45]:


newdata   # this is out dataset


# ### Spliting the data into x and y

# In[46]:


x=newdata.drop(columns=['Type'])
y=newdata['Type']


# In[47]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=4)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier    # Import Random Forest Classifier package

model=RandomForestClassifier(n_estimators=1000)
model.fit(xtrain,ytrain)
testpred=model.predict(xtest)
trainpred=model.predict(xtrain)



# In[ ]:


from sklearn.metrics import classification_report     # Import Classification Report package
print(classification_report(ytrain,trainpred))


# In[ ]:


from sklearn import metrics

cm=metrics.confusion_matrix(ytrain,trainpred)
cm=metrics.ConfusionMatrixDisplay(cm,display_labels=['fixed-price','hourly'])
print ('Confusion Matrix for Random Forest Training Prediction')
cm.plot()
plt.show()


# In[ ]:


print(classification_report(ytest,testpred))


# In[ ]:


cm=metrics.confusion_matrix(ytest,testpred)
cm=metrics.ConfusionMatrixDisplay(cm,display_labels=['fixed-price','hourly'])
print ('Confusion Matrix for Random Forest Test Prediction')
cm.plot()
plt.show()


# We can see with the help of confusion matrix that number of False Positive and False Negative is very low.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




