#!/usr/bin/env python
# coding: utf-8

# In[1]:


#For this kernel, I anlyze and pre-process wine's features on Red Wine Quality Dataset
#Then, six different classification models are build including Logistic Regression, K-Nearest Neighbor (KNN), Support Vector Machine (SVM), Naive Bayes, Decision Tree, Random Forest.
#Finally, visualizing the performence of the models.


# In[2]:


#Library
import pandas as pd
import numpy as np


# In[3]:


#Insert data
data_path = "Downloads\winequality-red.csv"
data = pd.read_csv(data_path)


# In[4]:


data.head()


# In[5]:


print('Data shape: ',data.shape)


# In[6]:


data.info()


# In[7]:


#Check the effect of properties to the wine's quality

data[["fixed acidity","quality"]].groupby(["quality"],as_index=False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[8]:


#the increase fixed acidicity positively affects the quality


# In[9]:


#volatile acidity
data[["volatile acidity","quality"]].groupby(["quality"],as_index=False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[10]:


#the degree of volatile acidicity positively affects quality


# In[11]:


#citric acid 
data[["citric acid","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[12]:


#the increase citric acid have positive effect to quality


# In[13]:


#residual sugar  
data[["residual sugar","quality"]].groupby(["quality"],as_index=False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[14]:


#there is no clue to see the relation between residual sugar and quality


# In[15]:


#chlorides
data[["chlorides","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[16]:


# There is negative effect of increasing chlorides amount on quality


# In[17]:


#free sulfur dioxide 
data[["free sulfur dioxide","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[18]:


#there is no clue to see the relation between free sulfur dioxide and quality


# In[19]:


#total sulfur dioxide 
data[["total sulfur dioxide","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[20]:


#density
data[["density","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[21]:


# There is slightly negative effect of increasing density on quality


# In[22]:


#pH
data[["pH","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[23]:


# There is negative effect of increasing pH amount on quality


# In[24]:


#sulphates 
data[["sulphates","quality"]].groupby(["quality"],as_index = False).mean().sort_values(by="quality").style.background_gradient("Reds")


# In[25]:


#Higher sulphates amount leads to higher wine's quality


# In[26]:


#alcohol
data[["alcohol","quality"]].groupby(["quality"],as_index = False).mean().sort_values("quality").style.background_gradient("Reds")


# In[27]:


#Higher alcohol amount leads to higher wine's quality


# In[28]:


#Visualization
#Have a look on features one by one
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


# In[29]:


#fixed acidity
quality = [3,4,5,6,7,8]
fixed_acidity_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x['fixed acidity'].mean()
    fixed_acidity_mean.append(mean)
    
plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
plt.hist(data["fixed acidity"],color = "red")
plt.xlabel("fixed acidity")
plt.ylabel("Frequency")
plt.title("fixed acidity histogram", color = "black", fontweight ="bold", fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["fixed acidity"],fit = norm, color= "red")
plt.title("fixed Acidity Distplot", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x=quality, y = fixed_acidity_mean, palette = "YlOrRd")
plt.title("the Average value of fixed acidity by quality", color = "black", fontweight = "bold", fontsize = 11)
plt.xlabel("quality")
plt.ylabel("fixed acidity")

plt.subplot(2,2,4)
sns.boxplot(data["quality"], data["fixed acidity"], palette = "YlOrRd")
plt.title("fixed acidity and quality", color = "black", fontweight = "bold", fontsize = 11)

plt.show()


# In[30]:


#Fixed acid have no effect on wine's quality
#According to graph 2, there is right skewness.


# In[31]:


#Volatile acidity
volatile_acidity_mean=[]

for each in quality:
    x=data[data["quality"]==each]
    mean = x["volatile acidity"].mean()
    volatile_acidity_mean.append(mean)


plt.figure(figsize = (15,10))

plt.subplot(2,2,1)
plt.hist(data["volatile acidity"],color = "orange")
plt.xlabel("Frequency")
plt.ylabel("volatile acidity")
plt.title("volatile acidity histogram", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["volatile acidity"], fit = norm, color = "orange")
plt.title("volatile acidity distplot", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x=quality, y = volatile_acidity_mean, palette = "YlOrBr")
plt.xlabel("wine quality")
plt.ylabel("volatile acidity")
plt.title("the average of volatile acidity by quality", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,4)
sns.boxplot(data["quality"], data["volatile acidity"], palette = "YlOrBr")
plt.title("volatile acidity and quality", color = "black", fontweight = "bold", fontsize = 11)

plt.show()


# In[32]:


#there is no skewness in distribution
#the decrease of volatic acidity positively affects wine quality


# In[33]:


#citric acid
citric_acid_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x["citric acid"].mean()
    citric_acid_mean.append(mean)
    
plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
plt.hist(data["citric acid"], color = "purple")
plt.xlabel("citric acid")
plt.ylabel("Frequecy")
plt.title("citric acid histogram", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["citric acid"], fit = norm, color="purple")
plt.title("citric acid distplot", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x=quality, y =citric_acid_mean, palette = "rocket")
plt.xlabel("quality")
plt.ylabel("citric acid")
plt.title("the average value of citric acid by quality", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,4)
sns.boxplot(data["quality"], data["citric acid"], palette = "rocket")
plt.title("citric acid and quality", color = "black", fontweight = "bold", fontsize = 11)

plt.show()


# In[34]:


#The increase of citric acid amount have positive effect to the wine quality


# In[35]:


#residual sugar
residual_sugar_mean = []

for each in quality:
    x=data[data["quality"]==each]
    mean = x["residual sugar"].mean()
    residual_sugar_mean.append(mean)
    
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.hist(data["residual sugar"], color = "lightgreen")
plt.xlabel("residual sugar")
plt.ylabel("Frequency")
plt.title("residual sugar histogram", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["residual sugar"], fit = norm, color = "lightgreen")
plt.title("residual sugar distplot", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x=quality, y = residual_sugar_mean, palette = "Greens")
plt.xlabel("quality")
plt.ylabel("residual sugar")
plt.title("the average value of residual sugar by quality", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,4)
sns.boxplot(data["quality"], data["residual sugar"], palette = "Greens")
plt.title("quality and residual sugar", color = "black", fontweight = "bold", fontsize = 11)

plt.show()


# In[36]:


#


# In[37]:


#chlorides
chlorides_mean = []

for each in quality:
    x=data[data["quality"]==each]
    mean = x["chlorides"].mean()
    chlorides_mean.append(mean)
    
plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
plt.hist(data["chlorides"],color="blue")
plt.xlabel("chlorides")
plt.ylabel("Frequency")
plt.title("chlorides histogram", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["chlorides"], fit = norm, color = "blue")
plt.title("clorides distplot", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x=quality, y= chlorides_mean, palette = "crest")
plt.xlabel("quality")
plt.ylabel("chlorides")
plt.title("the average value of chlorides by quality", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,4)
sns.boxplot(data["quality"], data["chlorides"], palette = "crest")
plt.title("quality and chlorides", color = "black", fontweight = "bold", fontsize = 11)

plt.show()




# In[38]:


#The decrease in chlorides have positive effect to wine's quality
#According to the fourth grapth, we have lots of outliers


# In[39]:


#free sulfur dioxide

free_sulfur_dioxide_mean = []

for each in quality:
    x=data[data["quality"]==each]
    mean = x["free sulfur dioxide"].mean()
    free_sulfur_dioxide_mean.append(mean)
    
plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
plt.hist(data["free sulfur dioxide"],color="brown")
plt.xlabel("free sulfur dioxide")
plt.ylabel("Frequency")
plt.title("free sulfur dioxide histogram", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["free sulfur dioxide"], fit = norm, color = "brown")
plt.title("free sulfur dioxide distplot", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x=quality, y= free_sulfur_dioxide_mean, palette = "Set2")
plt.xlabel("quality")
plt.ylabel("free sulfur dioxide")
plt.title("the average value of free sulfur dioxide by quality", color = "black", fontweight = "bold", fontsize = 11)

plt.subplot(2,2,4)
sns.boxplot(data["quality"], data["free sulfur dioxide"], palette = "Set2")
plt.title("quality and free sulfur dioxide", color = "black", fontweight = "bold", fontsize = 11)

plt.show()


# In[40]:


#there is no clear influence between free sulfur dioxide and wine's quality
#As we can see from the second graph, the graph is tailing to the right
#There is also multiple outliers in the forth grapth


# In[41]:


#total sulfur dioxide

total_sulfurdioxide_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x["total sulfur dioxide"].mean()
    total_sulfurdioxide_mean.append(mean)


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.hist(data["total sulfur dioxide"], color="#F1E68C")
plt.xlabel("total sulfur dioxide")
plt.ylabel("Frequency")
plt.title("total sulfur dioxide histogram", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["total sulfur dioxide"], fit=norm, color="#F5E68C")
plt.title("total sulfur dioxide Distplot", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x = quality, y = total_sulfurdioxide_mean, palette= "BrBG")
plt.title("the average value of total sulfur dioxide by quality", color = "black", fontweight= 'bold', fontsize = 11)
plt.xlabel("quality")
plt.ylabel("total sulfur dioxide mean")

plt.subplot(2,2,4)
sns.boxplot(data['quality'], data["total sulfur dioxide"], palette='BrBG')
plt.title("total sulfur dioxide & quality", color = "black", fontweight= 'bold', fontsize = 11)

plt.show()


# In[42]:


#There is skewness to the right


# In[43]:


#density
density_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x["density"].mean()
    density_mean.append(mean)


plt.figure(figsize=(13,10))
plt.subplot(2,2,1)
plt.hist(data["density"], color="lightblue")
plt.xlabel("density")
plt.ylabel("Frequency")
plt.title("density histogram", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["density"], fit=norm)
plt.title("density Distplot", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x = quality, y = density_mean, palette= "ocean")
plt.title("the average value of density by quality", color = "black", fontweight= 'bold', fontsize = 11)
plt.xlabel("quality")
plt.ylabel("density mean")

plt.subplot(2,2,4)
sns.boxplot(data['quality'], data["density"], palette='ocean')
plt.title("density & quality", color = "black", fontweight= 'bold', fontsize = 11)

plt.show()


# In[44]:


#The density feature has normal distribution and have no effect to wine quality
#There is many outliers need to be removed


# In[45]:


pH_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x["pH"].mean()
    pH_mean.append(mean)


plt.figure(figsize=(13,10))
plt.subplot(2,2,1)
plt.hist(data["pH"], color="#00FF00")
plt.xlabel("pH")
plt.ylabel("Frequency")
plt.title("pH histogram", color = "black", fontweight= 'bold', fontsize = 11)
plt.subplot(2,2,2)
sns.distplot(data["pH"], fit=norm, color = "#00FF00")
plt.title("pH Distplot", color = "black", fontweight= 'bold', fontsize = 11)
plt.subplot(2,2,3)
sns.barplot(x = quality, y = pH_mean, palette= "hsv")
plt.title("the average value of pH by quality", color = "black", fontweight= 'bold', fontsize = 11)
plt.xlabel("quality")
plt.ylabel("pH mean")
plt.subplot(2,2,4)
sns.boxplot(data['quality'], data["pH"], palette='hsv')
plt.title("pH & quality", color = "black", fontweight= 'bold', fontsize = 11)

plt.show()


# In[46]:


#The slight decrese of pH value positively affects wine's quality
#There are multiple outliers need to be removed


# In[47]:


#sulphates

sulphates_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x["sulphates"].mean()
    sulphates_mean.append(mean)


plt.figure(figsize=(13,10))
plt.subplot(2,2,1)
plt.hist(data["sulphates"], color="plum")
plt.xlabel("sulphates")
plt.ylabel("Frequency")
plt.title("sulphates histogram", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["sulphates"], fit=norm, color="plum")
plt.title("sulphates Distplot", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x = quality, y = sulphates_mean, palette= "twilight")
plt.title("the average value of sulphates by quality", color = "black", fontweight= 'bold', fontsize = 11)
plt.xlabel("quality")
plt.ylabel("sulphates mean")

plt.subplot(2,2,4)
sns.boxplot(data['quality'], data["sulphates"], palette='twilight')
plt.title("sulphates & quality", color = "black", fontweight= 'bold', fontsize = 11)

plt.show()


# In[48]:


#The second grapth is tailing to the right
#The increase of sulphates have positive effect to wine's quality
#There are many outliers need to be removed


# In[49]:


#alcohol

alcohol_mean = []

for each in quality:
    x = data[data["quality"] == each]
    mean = x["alcohol"].mean()
    alcohol_mean.append(mean)


plt.figure(figsize=(13,10))
plt.subplot(2,2,1)
plt.hist(data["alcohol"], color="#DAA520")
plt.xlabel("alcohol")
plt.ylabel("Frequency")
plt.title("alcohol histogram", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,2)
sns.distplot(data["alcohol"], fit=norm, color="#DAA520")
plt.title("alcohol Distplot", color = "black", fontweight= 'bold', fontsize = 11)

plt.subplot(2,2,3)
sns.barplot(x = quality, y = alcohol_mean, palette= "CMRmap")
plt.title("the average value of alcohol by quality", color = "black", fontweight= 'bold', fontsize = 11)
plt.xlabel("quality")
plt.ylabel("alcohol mean")

plt.subplot(2,2,4)
sns.boxplot(data['quality'], data["alcohol"], palette='CMRmap')
plt.title("alcohol & quality", color = "black", fontweight= 'bold', fontsize = 11)

plt.show()


# In[50]:


#There is right skewness
#The increase in alcohol positively affects to the wine's quality


# In[51]:


Number = data["quality"].value_counts().values
Label = data["quality"].value_counts().index
circle = plt.Circle((0,0), 0.3, color = "white")
explodeTuple = (0.0, 0.0, 0.0, 0.1, 0.2, 0.3)

plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
sns.countplot(data["quality"])
plt.xlabel("quality")
plt.title("wine quality distribution", color = "black", fontweight = "bold", fontsize =11)

plt.subplot(1,2,2)
plt.pie(Number, labels = Label, autopct = '%1.2f%%',explode = explodeTuple, startangle = 60)
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("quality distribution", color = "black", fontweight= 'bold', fontsize = 11)
plt.legend(bbox_to_anchor=(1, 1))

plt.show()


# In[52]:


#Skewness Correction
#fixed acidity
#free sulfur dioxide
#total sulfur dioxide
#alcohol


# In[53]:


#fixed acidity
(mu, sigma) = norm.fit(data["fixed acidity"])
print(" mu {} : {}, sigma {} : {}".format("fixed acidity", mu, "fixed acidity", sigma))


# In[54]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(data["fixed acidity"], fit = norm, color = "red")
plt.title("fixed acidity displot", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["fixed acidity"],plot = plt)
plt.show()


# In[55]:


from scipy.stats import boxcox


# In[56]:


data["fixed acidity"], lam_fixed_acidity = boxcox(data["fixed acidity"])


# In[57]:


(mu, sigma) = norm.fit(data["fixed acidity"])
print("mu {}: {}, sigma {}: {}".format("fixed acidity",mu,"fixed acidity", sigma))


# In[58]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(data["fixed acidity"], fit = norm, color = "red")
plt.subplot(1,2,2)
stats.probplot(data["fixed acidity"], plot = plt)
plt.show()


# In[59]:


#residual sugar
(mu,sigma) = norm.fit(data["residual sugar"])
print("mu {} : {}, sigma {}: {}".format("residual sugar", mu,"residual sugar", sigma))


# In[60]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(data["residual sugar"], fit = norm, color = "orange")
plt.title("residual sugar distplot", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["residual sugar"], plot = plt)
plt.show()


# In[61]:


data["residual sugar"], lam_residual_sugar = boxcox(data["residual sugar"])


# In[62]:


(mu,sigma) = norm.fit(data["residual sugar"])
print("mu {} : {}, sigma {}: {}".format("residual sugar", mu,"residual sugar", sigma))


# In[63]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(data["residual sugar"], fit = norm, color = "orange")
plt.title("residual sugar distplot", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["residual sugar"], plot = plt)
plt.show()


# In[64]:


#free sulfur dioxide
(mu,sigma) = norm.fit(data["free sulfur dioxide"])
print("mu {} : {}, sigma {}: {}".format("free sulfur dioxide", mu,"free sulfur dioxide", sigma))


# In[65]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(data["free sulfur dioxide"], fit = norm, color = "brown")
plt.title("free sulfur dioxide", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["free sulfur dioxide"], plot = plt)
plt.show()


# In[66]:


data["free sulfur dioxide"], lam_sulfur_dioxide = boxcox(data["free sulfur dioxide"])


# In[67]:


(mu,sigma) = norm.fit(data["free sulfur dioxide"])
print("mu {} : {}, sigma {}: {}".format("free sulfur dioxide", mu,"free sulfur dioxide", sigma))


# In[68]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(data["free sulfur dioxide"], fit = norm, color = "brown")
plt.title("free sulfur dioxide", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["free sulfur dioxide"], plot = plt)
plt.show()


# In[69]:


#total sulfur dioxide
(mu, sigma) = norm.fit(data["total sulfur dioxide"])
print("mu {} : {}, sigma {} : {}".format("total sulfur dioxide", mu, "total sulfur dioxide", sigma))


# In[70]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(data["total sulfur dioxide"], fit=norm, color="#F1E68C")
plt.title("total sulfur dioxide Distplot", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["total sulfur dioxide"], plot = plt)
plt.show()


# In[71]:


data["total sulfur dioxide"], lam_toal_sulphur_dioxide = boxcox(data["total sulfur dioxide"])


# In[72]:


(mu, sigma) = norm.fit(data["total sulfur dioxide"])
print("mu {} : {}, sigma {} : {}".format("total sulfur dioxide", mu, "total sulfur dioxide", sigma))


# In[73]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.distplot(data["total sulfur dioxide"], fit=norm, color="#F1E68C")
plt.title("total sulfur dioxide Distplot", color = "black", fontweight = "bold", fontsize = 11)
plt.subplot(1,2,2)
stats.probplot(data["total sulfur dioxide"], plot = plt)
plt.show()


# In[74]:


#alcohol
(mu, sigma) = norm.fit(data["alcohol"])
print("mu {}:{}, sigma {}: {}".format("alcohol", mu, "alcohol", sigma))


# In[75]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(data["alcohol"], fit = norm, color = "#DAA520")
plt.title("alcohol distplot", color = "black", fontweight = "bold", fontsize =11)
plt.subplot(1,2,2)
stats.probplot(data["alcohol"], plot = plt)
plt.show()


# In[76]:


data["alcohol"], lam_alcohol = boxcox(data["alcohol"])


# In[77]:


(mu, sigma) = norm.fit(data["alcohol"])
print("mu {}:{}, sigma {}: {}".format("alcohol", mu, "alcohol", sigma))


# In[78]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(data["alcohol"], fit = norm, color = "#DAA520")
plt.title("alcohol distplot", color = "black", fontweight = "bold", fontsize =11)
plt.subplot(1,2,2)
stats.probplot(data["alcohol"], plot = plt)
plt.show()


# In[79]:


#Outliers Detection
from collections import Counter


# In[80]:


def detect_outliers(df,features):
    outlier_indices= []
    
    for c in features:
        Q1 = np.percentile(df[c],25) #first quartile
        Q3 = np.percentile(df[c],75) #third quartile
        IQR = Q3 - Q1 #Interquartile range
        outlier_step = IQR *1.5
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c]> Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    return outlier_indices


# In[81]:


print("number of outliers detected: ", len(detect_outliers(data,data.columns[:-1])))
data.loc[detect_outliers(data,data.columns[:-1])]


# In[82]:


data = data.drop(detect_outliers(data, data.columns[:-1]), axis = 0).reset_index(drop = True)


# In[83]:


bins = [2,6.5,8]
labels = ["bad","good"]
data["quality"] = pd.cut(x=data["quality"], bins = bins, labels = labels)


# In[84]:


data["quality"].value_counts()


# In[85]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data["quality"] = labelencoder.fit_transform(data["quality"])
data.head()


# In[86]:


X = data.drop("quality", axis = 1 ).values
y = data["quality"].values.reshape(-1,1)


# In[87]:


#Train - Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 50)


# In[88]:


#Classfication Model
#Feature Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[89]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
lr = LogisticRegression(random_state = 50)
lr.fit(X_train_scaled, y_train.ravel())


# In[90]:


#Predicting Cross Validation Score
from sklearn.model_selection import cross_val_score
cv_lr = cross_val_score(estimator = lr, X = X_train_scaled, y=y_train)
print("CV: ", cv_lr.mean())

y_pred_lr_train = lr.predict(X_train_scaled)
accuracy_lr_train = accuracy_score(y_train, y_pred_lr_train)
print("Training set: ", accuracy_lr_train)

y_pred_lr_test = lr.predict(X_test_scaled)
accuracy_lr_test = accuracy_score(y_test, y_pred_lr_test)
print("Test set: ", accuracy_lr_test)


# In[91]:


confusion_matrix(y_test,y_pred_lr_test)


# In[92]:


tp_lr = confusion_matrix(y_test, y_pred_lr_test)[0,0]
fp_lr = confusion_matrix(y_test, y_pred_lr_test)[0,1]
fn_lr = confusion_matrix(y_test, y_pred_lr_test)[1,0]
tn_lr = confusion_matrix(y_test, y_pred_lr_test)[1,1]


# In[93]:


#K-Nearest Method
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train.ravel())


# In[94]:


#Predicting Cross Validation Score
cv_knn = cross_val_score(estimator = knn, X = X_train_scaled, y=y_train)
print("CV: ", cv_knn.mean())

y_pred_knn_train = knn.predict(X_train_scaled)
accuracy_knn_train = accuracy_score(y_train, y_pred_knn_train)
print("Training set: ", accuracy_knn_train)

y_pred_knn_test = knn.predict(X_test_scaled)
accuracy_knn_test = accuracy_score(y_test, y_pred_knn_test)
print("Test set: ", accuracy_knn_test)


# In[95]:


confusion_matrix(y_test,y_pred_knn_test)


# In[96]:


tp_knn = confusion_matrix(y_pred_knn_test, y_test)[0,0]
fp_knn = confusion_matrix(y_pred_knn_test, y_test)[0,1]
fn_knn = confusion_matrix(y_pred_knn_test, y_test)[1,0]
tn_knn = confusion_matrix(y_pred_knn_test, y_test)[1,1]


# In[97]:


#Support Vector Machine (SVM - linear)
from sklearn.svm import SVC 
svc = SVC(kernel = 'linear')
svc.fit(X_train_scaled, y_train.ravel())


# In[98]:


#Cross Validation
cv_svm = cross_val_score(estimator = svc, X = X_train_scaled, y=y_train)
print("CV: ", cv_svm.mean())

y_pred_svm_train = svc.predict(X_train_scaled)
accuracy_svm_train = accuracy_score(y_train, y_pred_svm_train)
print("Training set: ", accuracy_svm_train)


y_pred_svm_test = svc.predict(X_test_scaled)
accuracy_svm_test = accuracy_score(y_test, y_pred_svm_test)
print("Test set: ", accuracy_svm_train)


# In[99]:


confusion_matrix(y_test, y_pred_svm_test)


# In[100]:


tp_svm = confusion_matrix(y_test, y_pred_svm_test)[0,0]
fp_svm = confusion_matrix(y_test, y_pred_svm_test)[0,1]
fn_svm = confusion_matrix(y_test, y_pred_svm_test)[1,0]
tn_svm = confusion_matrix(y_test, y_pred_svm_test)[1,1]


# In[101]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_scaled, y_train.ravel())


# In[102]:


#Cross Validation
cv_nb = cross_val_score(estimator = nb, X = X_train_scaled, y = y_train)
print("CV: ", cv_nb.mean())

y_pred_nb_train = nb.predict(X_train_scaled)
accuracy_nb_train = accuracy_score(y_train, y_pred_nb_train)
print("Training set: ", accuracy_nb_train)

y_pred_nb_test = nb.predict(X_test_scaled)
accuracy_nb_test = accuracy_score(y_test, y_pred_nb_test)
print("Test set: ", accuracy_nb_test)


# In[103]:


confusion_matrix(y_test, y_pred_nb_test)


# In[104]:


tp_nb = confusion_matrix(y_test, y_pred_svm_test)[0,0]
fp_nb = confusion_matrix(y_test, y_pred_svm_test)[0,1]
fn_nb = confusion_matrix(y_test, y_pred_svm_test)[1,0]
tn_nb = confusion_matrix(y_test, y_pred_svm_test)[1,1]


# In[105]:


#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 50)
dt.fit(X_train_scaled, y_train.ravel())


# In[106]:


#Cross Validation
cv_dt = cross_val_score(estimator = dt, X=X_train_scaled, y=y_train)
print("CV: ", cv_dt.mean())

y_pred_dt_train = dt.predict(X_train_scaled)
accuracy_dt_train = accuracy_score(y_train, y_pred_dt_train)
print("Traning set: ", accuracy_dt_train)

y_pred_dt_test = dt.predict(X_test_scaled)
accuracy_dt_test = accuracy_score(y_test, y_pred_dt_test)
print("Test set: ", accuracy_dt_test)


# In[107]:


confusion_matrix(y_test, y_pred_dt_test)


# In[108]:


tp_dt = confusion_matrix(y_test, y_pred_dt_test)[0,0]
fp_dt = confusion_matrix(y_test, y_pred_dt_test)[0,1]
fn_dt = confusion_matrix(y_test, y_pred_dt_test)[1,0]
tn_dt = confusion_matrix(y_test, y_pred_dt_test)[1,1]


# In[109]:


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 50)
rf.fit(X_train_scaled, y_train.ravel())


# In[110]:


#Cross Validation
cv_rf = cross_val_score(estimator = rf, X=X_train_scaled, y= y_train)
print("CV: ", cv_rf.mean())

y_pred_rf_train = rf.predict(X_train_scaled)
accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
print("CV: ", accuracy_rf_train)

y_pred_rf_test = rf.predict(X_test_scaled)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
print("CV: ", accuracy_rf_test)


# In[111]:


confusion_matrix(y_test, y_pred_rf_test)


# In[112]:


tp_rf = confusion_matrix(y_test, y_pred_rf_test)[0,0]
fp_rf = confusion_matrix(y_test, y_pred_rf_test)[0,1]
fn_rf = confusion_matrix(y_test, y_pred_rf_test)[1,0]
tn_rf = confusion_matrix(y_test, y_pred_rf_test)[1,1]


# In[113]:


#Measuring the error:
models = [('Logistic Regression', tp_lr, fp_lr, fn_lr, tn_lr, accuracy_lr_train, accuracy_lr_test, cv_lr.mean()),
          ('K-Nearest Neighbor (KNN)', tp_knn, fp_knn, fn_knn, tn_knn, accuracy_knn_train, accuracy_knn_test, cv_knn.mean()),
          ('SVM (linear)', tp_svm, fp_svm, fn_svm, tn_svm, accuracy_svm_train, accuracy_svm_test, cv_svm.mean()),
          ('Naive Bayes', tp_nb, fp_nb, fn_nb, tn_nb, accuracy_nb_train, accuracy_nb_test, cv_nb.mean()),
          ('Decision Tree Classification', tp_dt, fp_dt, fn_dt, tn_dt, accuracy_dt_train, accuracy_dt_test, cv_dt.mean()),
          ('Random Forest Tree Classification', tp_rf, fp_rf, fn_rf, tn_rf, accuracy_rf_train, accuracy_rf_test, cv_rf.mean())
         ]


# In[114]:


summary = pd.DataFrame(data= models, columns = ['Model', 'True Possitive', 'False Positive', 'False Negative', 'True Negative', 'Accuracy Score (Train)', 'Accuracy Score (Test)','Cross Validation'])
summary


# In[115]:


#Visualize performence

f, axe = plt.subplots(1,1,figsize = (10,3))
summary.sort_values(by=['Cross Validation'], ascending = False, inplace = True)

sns.barplot(x='Cross Validation', y = 'Model', data=summary, ax = axe)
axe.set_xlabel('Cross Validation Score', size = 10)
axe.set_ylabel('Model',size = 10)
axe.set_xlim(0,1.0)
plt.show()


# In[116]:


f, axes = plt.subplots(2,1,figsize = (10,10))
summary.sort_values(by = ['Accuracy Score (Train)'], ascending = False, inplace = True)

sns.barplot(x='Accuracy Score (Train)', y = 'Model', data = summary, palette = 'Blues_d', ax = axes[0])
axes[0].set_xlabel('Accuracy Score (Train)', size = 10)
axes[0].set_ylabel('Models',size = 10)
axes[0].set_xlim(0, 1.0)

summary.sort_values(by = ['Accuracy Score (Test)'], ascending = False, inplace = True)
sns.barplot(x='Accuracy Score (Test)', y='Model', data = summary, palette = 'Reds_d', ax=axes[1])
axes[1].set_xlabel('Accuracy Score (Test)', size = 10)
axes[1].set_ylabel('Models',size = 10)
axes[1].set_xlim(0, 1.0)

plt.show()


# In[117]:


summary.sort_values(by=(['Accuracy Score (Test)']), ascending = True, inplace = True)

f,axe = plt.subplots(1,1, figsize = (15,7))
sns.barplot(x=summary['Model'],y=summary['False Negative'] + summary['False Positive'], ax=axe)
axe.set_xlabel('Model', size = 16)
axe.set_ylabel('False Observation' , size = 16)

plt.show()


# #According to accuracy score on training set, Cross Validation and Test set, Random Forest Tree did a best job on Wine qualify classification
# 
# 
