#!/usr/bin/env python
# coding: utf-8

# # LIBRARIES 

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as npy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mat_plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance


# # 

# # IMPORTING DATASET

# In[3]:


#Read dataset
dataSet = pd.read_csv('CICIDS2017_sample.csv') 
#The data is visualized based on the attacks 
# The results in this code is based on the original CICIDS2017 dataset. Please go to cell [21] if you work on the sampled dataset. 


# In[4]:


dataSet


# In[5]:


#To get the count of attacks in the dataset
dataSet.Label.value_counts()


# # 

# # VISUALIZATION OF DATA 

# In[6]:


#Different types of attacks present in the Dataset 
labels=['BENIGN', 'DoS','PortScan','BruteForce', 'WebAttack','Bot','Infiltration']     
#print(labels)
#The Total Count of each individual attack
values=list(dataSet.Label.value_counts())
#Plot Figure is used to display the data based on the given size
mat_plt.figure(figsize=(9,6))
#To represent the data in the form of pie
mat_plt.pie(values,labels=labels,autopct='%.2f%%',shadow=True)
mat_plt.show()

#Bar Diagram
x = npy.array(list(labels)) 
y = npy.array(list(dataSet.Label.value_counts())) 
fig, ax = mat_plt.subplots() 
ax.bar(x, y)
ax.set_title("Example Bar Chart") 
ax.set_xlabel("Category") 
ax.set_ylabel("Value") 
mat_plt.show()


# # 

# # DATA PREPROCESSING

# In[7]:


# Z-score normalization 
# Here we used Z-score inorder to remove the outliers in the dataset and to normalize the features into similar scale
features = dataSet.dtypes[dataSet.dtypes != 'object'].index
dataSet[features] = dataSet[features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# Fill empty values by 0
dataSet = dataSet.fillna(0)


# In[8]:


#LabelEncoder is used to convert categorical(non-numeric) data into numeric values.
labelencoder = LabelEncoder()
#The Attacks such as 'BENIGN', 'DoS','PortScan','BruteForce', 'WebAttack','Bot' and 'Infiltration' converted into numeric values based on the alphabetical order
dataSet.iloc[:, -1] = labelencoder.fit_transform(dataSet.iloc[:, -1])


# In[9]:


#The count of attacks are found and assigned to respective attacks in numeric form
dataSet.Label.value_counts()


# In[10]:


# retain the minority class instances and sample the majority class instances
df_minor = dataSet[(dataSet['Label']==6)|(dataSet['Label']==1)|(dataSet['Label']==4)]
df_major = dataSet.drop(df_minor.index)


# In[11]:


X = df_major.drop(['Label'],axis=1) 
y = df_major.iloc[:, -1].values.reshape(-1,1)
print(y)
y=npy.ravel(y)
print(y)


# In[12]:


# use k-means to cluster the data samples and select a proportion of data from each cluster
from sklearn.cluster import MiniBatchKMeans
k_means = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)


# In[13]:


klabel=k_means.labels_
df_major['klabel']=klabel


# In[14]:


df_major['klabel'].value_counts()


# In[15]:


cols = list(df_major)
cols.insert(78, cols.pop(cols.index('Label')))
df_major = df_major.loc[:, cols]


# In[16]:


df_major


# In[17]:


def typicalSampling(group):
    name = group.name
    frac = 0.008
    return group.sample(frac=frac)

result = df_major.groupby(
    'klabel', group_keys=False
).apply(typicalSampling)


# In[18]:


result['Label'].value_counts()


# In[19]:


result


# In[20]:


result = result.drop(['klabel'],axis=1)
result = result.append(df_minor)


# In[21]:


result.to_csv('CICIDS2017_sample_km.csv',index=0)


# In[22]:


# Read the sampled dataset
dataSet=pd.read_csv('CICIDS2017_sample_km.csv')


# In[23]:


X = dataSet.drop(['Label'],axis=1).values
y = dataSet.iloc[:, -1].values.reshape(-1,1)
y=npy.ravel(y)


# # 

# # SPLIT DATA FOR TRAIN AND TEST

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# # 

# # FEATURE ENIGNEERING

# In[25]:


from sklearn.feature_selection import mutual_info_classif
importances = mutual_info_classif(X_train, y_train)


# In[26]:


# calculate the sum of importance scores
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
featureScaling = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    featureScaling.append(f_list[i][1])


# In[27]:


# select the important features from top to bottom until the accumulated importance reaches 90%
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
featureScaling = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    featureScaling.append(f_list2[i][1])
    if Sum2>=0.9:
        break        


# In[28]:


X_fs = dataSet[featureScaling].values


# In[29]:


X_fs


# In[30]:


X_fs.shape


# In[31]:


from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
fcbf = FCBFK(k = 20)
#fcbf.fit(X_fs, y)


# In[32]:


X_fss = fcbf.fit_transform(X_fs,y)


# In[33]:


X_fss.shape


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X_fss,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# In[35]:


X_train.shape


# In[36]:


pd.Series(y_train).value_counts()


# In[37]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000,4:1000})


# In[38]:


X_train, y_train = smote.fit_resample(X_train, y_train)


# In[39]:


pd.Series(y_train).value_counts()


# ## Learning Models of Machine Learning

# ### K Neighbors Classifier :

# In[40]:


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 3) 
classifier.fit(X_train,y_train)
#It is used to calculate the accuracy of the model. 1 represents predicted all the target labels correctly and 0 represents inaccurate.
classifier_score=classifier.score(X_test,y_test)

#It is used to make predictions using a trained classifier model on a new set of data represented by the features X_test. It identifies whether the data point belongs to classifier or not.
y_predict = classifier.predict(X_test)

#It is used to evaluate the performance of the classifier and determine which types of errors the classifier is making.
confusion_matrix_KNN = confusion_matrix(y_test, y_predict)
y_true=y_test
print('Accuracy of KNN: '+ str(classifier_score))
#It is used to compute the precision, recall, and F1 score for a classifier's predictions on a new set of data, where y_true are the true labels and y_predict are the predicted labels.
precision_KNN,recall_KNN,fscore_KNN,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of KNN: '+(str(precision_KNN)))
print('Recall of KNN: '+(str(recall_KNN)))
print('F1-score of KNN: '+(str(fscore_KNN)))
print("Accuracy_score :")
print(classification_report(y_test, y_predict))
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(confusion_matrix_KNN,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
mat_plt.xlabel("y_predict")
mat_plt.ylabel("y_true")
mat_plt.show()


# ### Support Vector Machine Classifier :

# In[41]:


# Importing SVC Classifer
from sklearn.svm import SVC
# Creates instance of the Support Vector Machine (SVM) classifier using a linear kernel
clf = SVC(kernel='linear')
# Fit: Used to train the model by fitting it to the training data
clf.fit(X_train,y_train)
# Returns the mean accuracy of the SVM classifier on the test data
clf_score=clf.score(X_test,y_test)
# Predict method applies the trained SVM classifier to the test data X_test and predicts the target labels for each sample in X_test
y_predict = clf.predict(X_test)
y_true=y_test
print('Accuracy of SVM: '+ str(clf_score))
precision_SVM,recall_SVM,fscore_SVM,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of SVM: '+(str(precision_SVM)))
print('Recall of SVM: '+(str(recall_SVM)))
print('F1-score of SVM: '+(str(fscore_SVM)))
# To evaluate the performance of a classification model and to identify any imbalances in the distribution of the target classes
print(classification_report(y_true,y_predict))
# Evaluate the accuracy of a classification model and to identify which classes are being misclassified
cm_SVM=confusion_matrix(y_true,y_predict)
f,ax=mat_plt.subplots(figsize=(5,5))
# Heatmaps are often used to visualize the results of statistical analyses or machine learning algorithms, such as the confusion matrix in a classification problem
sns.heatmap(cm_SVM,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax) 
mat_plt.xlabel("y_predict")
mat_plt.ylabel("y_true")
mat_plt.show()


# ### XGB Classifier :

# In[42]:


xg = xgb.XGBClassifier(n_estimators = 10)# Creates an instance of the XGBClassifier class from the XGBoost library with 10 decision trees as estimators. 
xg.fit(X_train,y_train) # Fit: Used to train the model by fitting it to the training data 
xg_score=xg.score(X_test,y_test)# Returns the mean accuracy of the XGB classifier on the test data
y_predict=xg.predict(X_test)# Predict method applies the trained XGB classifier to the test data X_test and predicts the target labels for each sample in X_test 
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision_XGB,recall_XGB,fscore_XGB,none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of XGBoost: '+(str(precision_XGB)))
print('Recall of XGBoost: '+(str(recall_XGB)))
print('F1-score of XGBoost: '+(str(fscore_XGB)))
print(classification_report(y_true,y_predict))# To evaluate the performance of a classification model  and find any imbalances 
cm_XGB=confusion_matrix(y_true,y_predict)#To evaluate accuracy of the classification model
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(cm_XGB,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)# To visualize Confusion Matrix
mat_plt.xlabel("y_pred")
mat_plt.ylabel("y_true")
mat_plt.show()


# ## Hyperparameter optimization (HPO) of XGBoost using Bayesian optimization with tree-based Parzen estimator (BO-TPE)

# In[43]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'learning_rate':  abs(float(params['learning_rate'])),

    }
    # Creating the XGBoost classifier
    clf = xgb.XGBClassifier( **params)
    # Fitting the classifier on the training set
    clf.fit(X_train, y_train)
    # Predicting on the test set
    y_pred = clf.predict(X_test)
    # Calculating the accuracy score of the predicted values
    score = accuracy_score(y_test, y_pred)

    # Returning the loss and status of the model
    return {'loss':-score, 'status': STATUS_OK }

# Defining the search space for the hyperparameters
space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
    'max_depth': hp.quniform('max_depth', 4, 100, 1),
    'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
}
# Using TPE algorithm to minimize the objective function and finding the best hyperparameters
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)
# Printing the best hyperparameters found by the algorithm
print("XGBoost: Hyperopt estimated optimum {}".format(best))


# In[44]:


xg = xgb.XGBClassifier(learning_rate= 0.7340229699980686, n_estimators = 70, max_depth = 14)
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
mat_plt.xlabel("y_pred")
mat_plt.ylabel("y_true")
mat_plt.show()


# In[45]:


xg_train=xg.predict(X_train)
xg_test=xg.predict(X_test)


# ### Random Forest Classifier :

# In[46]:


# Initializing the random forest classifier
rf = RandomForestClassifier(random_state = 0)
#Fitting the training data into the random forest classifier
rf.fit(X_train,y_train) 
# Scoring the accuracy of the random forest classifier
rf_score=rf.score(X_test,y_test)
# Predicting the test set using the fitted random forest classifier
y_predict=rf.predict(X_test)
# Getting the true labels of the test set
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
# It is used to compute the precision, recall, and F1 score for a classifier's predictions on a new set of data, where y_true are the true labels and y_predict are the predicted labels.
precision_RFC,recall_RFC,fscore_RFC,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
# Printing the precision, recall, and F1-score of the random forest classifier
print('Precision of RF: '+(str(precision_RFC)))
print('Recall of RF: '+(str(recall_RFC)))
print('F1-score of RF: '+(str(fscore_RFC)))
# Printing the classification report of the random forest classifier
print(classification_report(y_true,y_predict))
# It is used to evaluate the performance of the classifier and determine which types of errors the classifier is making.
confusion_matrix_RFC=confusion_matrix(y_true,y_predict)
# Plotting the confusion matrix using seaborn and matplotlib
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(confusion_matrix_RFC,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
mat_plt.xlabel("y_pred")
mat_plt.ylabel("y_true")
mat_plt.show()


# In[47]:


rf_hpo = RandomForestClassifier(n_estimators = 71, min_samples_leaf = 1, max_depth = 46, min_samples_split = 9, max_features = 20, criterion = 'entropy')
rf_hpo.fit(X_train,y_train)
rf_score=rf_hpo.score(X_test,y_test)
y_predict=rf_hpo.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
mat_plt.xlabel("y_pred")
mat_plt.ylabel("y_true")
mat_plt.show()


# In[48]:


rf_train=rf_hpo.predict(X_train)
rf_test=rf_hpo.predict(X_test)


# ### Decision Tree Classifier :

# In[49]:


dt = DecisionTreeClassifier(random_state = 0)# Create an instance of Decision Tree Classifier with random state set to 0 
dt.fit(X_train,y_train)# Fit: Used to train the model by fitting it to the training data 
dt_score=dt.score(X_test,y_test)#  Returns the mean accuracy of the DT classifier on the test data 
y_predict=dt.predict(X_test)# Predict method applies the trained DT classifier to the test data X_test and predicts the target labels for each sample in X_test 
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision_DT,recall_DT,fscore_DT,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of DT: '+(str(precision_DT)))
print('Recall of DT: '+(str(recall_DT)))
print('F1-score of DT: '+(str(fscore_DT)))
print(classification_report(y_true,y_predict))# To evaluate the performance of a classification model  and find any imbalances 
cm_DT=confusion_matrix(y_true,y_predict)# To evaluate the accuracy of the classification model
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(cm_DT,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)# To visualize Confusion Matrix
mat_plt.xlabel("y_pred")
mat_plt.ylabel("y_true")
mat_plt.show()


# In[50]:


dt_hpo = DecisionTreeClassifier(min_samples_leaf = 2, max_depth = 47, min_samples_split = 3, max_features = 19, criterion = 'gini')
dt_hpo.fit(X_train,y_train)
dt_score=dt_hpo.score(X_test,y_test)
y_predict=dt_hpo.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=mat_plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
mat_plt.xlabel("y_pred")
mat_plt.ylabel("y_true")
mat_plt.show()


# In[51]:


dt_train=dt_hpo.predict(X_train)
dt_test=dt_hpo.predict(X_test)

