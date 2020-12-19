# -*- coding: utf-8 -*-

"""
Created on Sun Jul  5 11:15:51 2020
@author: Alexandra Tzilivaki
"""
"""
1) shuffle the data / make final preprocessing calculate class weights
2) choose and test the performance of multiple classification models
3) choose the best -->RF
4) Train and test the RF on the stained neurons using cross validation suitable for multiclass
5) Evaluate the models to be used 
6) Predict the staining for the remaining cells
7) Check other analyses
# """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from itertools import cycle
from sklearn.preprocessing import label_binarize
import seaborn as sns
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, auc, roc_curve, make_scorer
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
# from imblearn.over_sampling import SMOTE, SMOT

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
                  
#################################################################################################################
#                              load data, shuffle etc
#################################################################################################################

data=pd.read_csv('finaldatadecember2020.csv', index_col='CellID')

data=shuffle(data) #suffling the data from the beginning!

dataUnknown=data.loc[data.staining =='Unknown']

del(dataUnknown['staining'])

dataknown=data.loc[data.staining != 'Unknown']
labels = {'WFS1': 0,'Reelin': 1, 'PV':2} 
dataknown.staining = [labels[item] for item in dataknown.staining] 


y=dataknown.staining

X=dataknown.iloc[:,:-1]
X=preprocessing.MinMaxScaler().fit_transform(X)

#here I tried other preprocessing methods.
# X=preprocessing.StandardScaler().fit_transform(X)
# X=preprocessing.Normalizer().fit_transform(X.T)
# X=X.T


##################################################################################################################
#       The dataset is multiclass (WFS1, Reelin, PV) and very imbalanced --vast majority WFS1.--
#       Reelin and WFS1 are very similar and we have only 7 PV  out of 272 labeled data. so.......
#       ----------------------------------- Class weight calculation!---------------------------------------------
##################################################################################################################

class_weights=compute_class_weight('balanced', np.unique(y), y)
print (class_weights)

labels=np.unique(y)

weights={}
for classes in labels: weights[classes]=class_weights[classes]





########################################################################
#                define train and evaluate multiple models
#                 Model selection! 
########################################################################
""" Gridsearch cv for multiple models in order to identify their best parameters"""

# Xgrid_train, Xgrid_test, ygrid_train, ygrid_test = train_test_split(X,y,train_size=0.80,test_size=0.20)


#Models creation
NB   = ComplementNB()
KNN   = KNeighborsClassifier()
# SVC   = SVC(gamma='auto',class_weight=weights, decision_function_shape='ovo')
RF    = RandomForestClassifier(class_weight=(weights))
LR    =  LogisticRegression()
MLP   = MLPClassifier()
model = [RF]  #example with RF change accordingly

#1. Complement Naive Bayes
NB_grid = dict()

#2. K-Nearest - Neighborg
n_neighbors = range(5,10)
metric = ['euclidean', 'manhattan', 'minkowski']
KNN_grid = dict(n_neighbors=n_neighbors,metric=metric)

#3. Support Vector Classifier
kernel = ['poly', 'rbf', 'sigmoid']
SVC_grid = dict(kernel=kernel)


#4. Random Forest
n_estimators = [10, 100]
RF_grid = dict(n_estimators=n_estimators)

#5. Logistic Regrresion
solvers = ['newton-cg', 'lbfgs', 'liblinear']
LR_grid = dict(solver=solvers, class_weight=weights)

#6. MLP
activation = ['relu']
max_iter=[100,1000,10]
hidden_layer_sizes = [(20,50),(50,20),(100,100),(200,2)]
MLP_grid=dict(activation=activation, max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)

#example with RF . change accordingly.
grids = [RF_grid]
scorer = make_scorer(f1_score,average = 'macro')

col = 0
for ind in range(0,len(model)):
    print(ind)
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(estimator=model[ind], 
                  param_grid=grids[ind], cv=cv,  
                  scoring= scorer,error_score=0)
    grid_clf_acc = grid_search.fit(X,y)

grid_search.best_params_


""" find the best models with cross validation and statification"""
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = StratifiedKFold(n_splits=5, shuffle=True) #no 10 cross since we have only 7 PV.
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1) #rmse
    return scores

# define models to test
def get_models():
    
    models, names = list(), list()
    
    
    # SVM include class weights. decision function suitable for multiclass
    models.append(SVC(gamma='auto', class_weight=weights, decision_function_shape='ovo')) #linear poly 
    names.append('SVM')
    
   
    # KNN. best performance 
    models.append(KNeighborsClassifier(metric='euclidean',n_neighbors=5))
    names.append('KNN')
    
    # multi layer perceptron(neural network.)
    models.append(MLPClassifier(hidden_layer_sizes=(100,100), activation='relu',max_iter=1000))
    names.append('MLP')
    
    # Random Forest. include class weights
    models.append(RandomForestClassifier(n_estimators=100, class_weight=weights, n_jobs=-1))
    names.append('RF')
    
    # Naive bayes. complement version is suitable for fulticlass
    models.append(ComplementNB())
    names.append('NB')
    
    # Logistic Regression, include class weights
    models.append(LogisticRegression(class_weight=(weights), solver='liblinear'))
    names.append('LR')
    
    
    return models, names



    # define models
   
models, names = get_models()
results = list()
    
    # evaluate each model
    
# for trials in range(15):

for i in range(len(models)):
   
    # evaluate the model and store results
    scores = evaluate_model(X, y, models[i])
    results.append(scores)
    # summarize performance
    
    print('>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))

# plot the results
plt.boxplot(results, labels=names, showmeans=True)
plt.show()



# ##############################################################################################################
# #                               choose the best model(s)
# #                          RF and MLP since they have the best accuracy!
# ###############################################################################################################


####### cross validation suitable for mutliclass ################

cv = StratifiedKFold(n_splits=5,shuffle=True) #random state default.  no 10 kfold since we have only 7 PV cells!!




modelrf = RandomForestClassifier(n_estimators=100, class_weight=weights, n_jobs=-1)

scoresrocaucrf=cross_val_score(modelrf, X,y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)
scoresaccuracyrf=cross_val_score(modelrf, X,y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean ROC AUC: %.3f' % np.mean(scoresrocaucrf))
print('Mean accuracy score: %.3f' % np.mean(scoresaccuracyrf))



modelmlp=MLPClassifier(hidden_layer_sizes=(100,100), activation='relu',max_iter=1000)

scoresrocaucmlp=cross_val_score(modelmlp, X,y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)
scoresaccuracymlp=cross_val_score(modelmlp, X,y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean ROC AUC: %.3f' % np.mean(scoresrocaucmlp))
print('Mean accuracy score: %.3f' % np.mean(scoresaccuracymlp))

######################### Train test and Exchaustively check the performance of the two best models###########

performancerf=pd.DataFrame()
accuracyrf=[]
balanced_accuracyrf=[]
precisionrf=[]
recallrf=[]
f1scorerf=[]
rocaucrf=[]


performancemlp=pd.DataFrame()
accuracymlp=[]
balanced_accuracymlp=[]
precisionmlp=[]
recallmlp=[]
f1scoremlp=[]
rocaucmlp=[]



for train_index, test_index in cv.split(X,y):

    print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    # sm = SMOTE(random_state=1)
    # x_train, y_train = sm.fit_resample(x_train,y_train)
    
#........................................................................
#.....................................................................
#....................RANDOM FOREST TRAIN FIT

    modelrf.fit(x_train,y_train)
    y_predrf=modelrf.predict(x_test)
    
    #probability of confidence let's say 99% sure to put it on class 2 
    #or 34% for 0 4%for 2 and the remaining for 1 . so it goes to 1
    y_predrocrf=modelrf.predict_proba(x_test) 
    
    # Binarize the output in order to calculate and plot the roc auc scores for each class.
    y_test2 = label_binarize(y_test, classes=[0, 1, 2])
    y_pred2rf=label_binarize(y_predrf, classes=[0,1,2])
    n_classesrf = y_test2.shape[1]
    y_scorerf=y_pred2rf
  
    feature_importancesrf=pd.Series(modelrf.feature_importances_, index=data.columns[0:21])
    feature_importancesrf=feature_importancesrf.sort_values()
  
    # plt.figure()
    # feature_importancesrf.plot(kind="barh")
    # plt.title("random forest")

  
    
    accuracyrf.append(accuracy_score(y_test,y_predrf))
   
    balanced_accuracyrf.append(balanced_accuracy_score(y_test,y_predrf)) #shall I put sample weights?
    
    precisionrf.append(precision_score(y_test, y_predrf, average='macro'))
    
    recallrf.append(recall_score(y_test, y_predrf, average='macro'))
    
    f1scorerf.append(f1_score(y_test, y_predrf, average='macro'))
   
    rocaucrf.append(roc_auc_score(y_test, y_predrocrf, multi_class='ovo',labels=labels,average='macro',max_fpr=1))


#---------------------------------------------------------------------
#                           MLP classifier--------------------------
#-------------------------------------------------------------------



    modelmlp.fit(x_train,y_train)
    y_predmlp=modelmlp.predict(x_test)
    
    #probability of confidence let's say 99% sure to put it on class 2 
    #or 34% for 0 4%for 2 and the remaining for 1 . so it goes to 1
    y_predrocmlp=modelmlp.predict_proba(x_test) 
    
    # Binarize the output in order to calculate and plot the roc auc scores for each class.
    y_test2 = label_binarize(y_test, classes=[0, 1, 2])
    y_pred2mlp=label_binarize(y_predmlp, classes=[0,1,2])
    n_classesmlp = y_test2.shape[1]
    y_scoremlp=y_pred2mlp
  

  
    accuracymlp.append(accuracy_score(y_test,y_predmlp))
   
    balanced_accuracymlp.append(balanced_accuracy_score(y_test,y_predmlp)) #shall I put sample weights?
    
    precisionmlp.append(precision_score(y_test, y_predmlp, average='macro'))
    
    recallmlp.append(recall_score(y_test, y_predmlp, average='macro'))
    
    f1scoremlp.append(f1_score(y_test, y_predmlp, average='macro'))
   
    rocaucmlp.append(roc_auc_score(y_test, y_predrocmlp, multi_class='ovo',labels=labels,average='macro',max_fpr=1))






performancerf['Accuracy']=accuracyrf
performancerf['Bal Accuracy']=balanced_accuracyrf
performancerf['Precision']=precisionrf
performancerf['Recall']=recallrf
performancerf['f1score']=f1scorerf
performancerf['ROCAUC']=rocaucrf


performancerf.loc['mean'] = performancerf.mean()
performancerf.loc['std'] = performancerf.std()


# sns.boxplot(x="variable", y="value", data=pd.melt(performancerf),palette="Set3").set_title('Random Forest Model')
# plt.savefig('rf_perfdec.eps')

performancemlp['Accuracy']=accuracymlp
performancemlp['Bal Accuracy']=balanced_accuracymlp
performancemlp['Precision']=precisionmlp
performancemlp['Recall']=recallmlp
performancemlp['f1score']=f1scoremlp
performancemlp['ROCAUC']=rocaucmlp

performancemlp.loc['mean'] = performancemlp.mean()
performancemlp.loc['std'] = performancemlp.std()

# sns.boxplot(x="variable", y="value", data=pd.melt(performancemlp),palette="Set3").set_title('Multi layer Perceptron Model')
# plt.savefig('mlp_perfdec.eps')

#///////////////////////////////////////////////////////////////////////////////////////




# totalperformance=pd.concat([performancerf,performancemlp],ignore_index=True)
# totalperformance['model']=['rf','rf','rf','rf','rf','mlp','mlp','mlp','mlp','mlp']


#######################################################################
#####################################################################
#############################################################################


# make actual predictions on nonstained cells. be nice please!

y_train2=dataknown.staining
x_train2=dataknown.iloc[:,:-1]
x_train2=preprocessing.MinMaxScaler().fit_transform(x_train2)
x_test2=dataUnknown.iloc[:,:]
# y_pred=dataunknown.staining
x_test2=preprocessing.MinMaxScaler().fit_transform(x_test2)

y_predrf=pd.DataFrame()

y_predmlp=pd.DataFrame()


model3 = RandomForestClassifier(n_estimators=100, class_weight=weights, n_jobs=-1)
model4 = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu',max_iter=1000)

for i in range(21):

    model3.fit(x_train2,y_train2)
    y_predrf[i]=model3.predict(x_test2)
    
    model4.fit(x_train2,y_train2)
    y_predmlp[i]=model4.predict(x_test2)

y_predrf=y_predrf.mode(axis=1)
y_predmlp=y_predmlp.mode(axis=1)

y_predrf=y_predrf.set_index(dataUnknown.index)
y_predmlp=y_predmlp.set_index(dataUnknown.index)


predictions=pd.concat([y_predmlp.rename(columns={0:'MLP'}),y_predrf.rename(columns={0:'RF'})],axis=1)
predictions['mismatch'] = np.where(y_predrf == y_predmlp, 'True', 'False')
predictions2=predictions.set_index(dataUnknown.index)

pred2=predictions2[predictions2.mismatch == 'False']

dataUnknown2=dataUnknown[~dataUnknown.index.isin(pred2.index)]

predictionsfinal=predictions2[~predictions2.index.isin(pred2.index)]
del(predictionsfinal['mismatch'])
del(predictionsfinal['MLP'])
dataUnknown2['staining']=predictionsfinal


alldatafinal=pd.concat([dataknown,dataUnknown2])

labelsfinal = { 0:'WFS1', 1:'Reelin', 2:'PV'} 
alldatafinal.staining = [labelsfinal[item] for item in alldatafinal.staining] 

my_pal = {"WFS1": "dimgrey", "Reelin": "steelblue", "PV":"darkred"}


v=sns.boxplot(x="staining",y="Halfwidth",data=alldatafinal,palette=(my_pal)).set_title('PaS neurons')

# labelspred = { 0:00, 1:11, 2:22} 


# alldatafinal.staining = [labelsfinal[item] for item in alldatafinal.staining] 

dfWFS1=alldatafinal.loc[alldatafinal['staining'] == 'WFS1']
dfReelin=alldatafinal.loc[alldatafinal['staining'] == 'Reelin']
dfPV=alldatafinal.loc[alldatafinal['staining'] == 'PV']  

del(dfReelin['staining'])


qrel = dfReelin.quantile([.25, .75])
qrel=qrel.append(dfReelin.median(), ignore_index=True)

del(dfPV['staining'])


qPV = dfPV.quantile([.25, .75])
qPV=qPV.append(dfPV.median(), ignore_index=True)


del(dfWFS1['staining'])


qw = dfWFS1.quantile([.25, .75])
qw=qw.append(dfWFS1.median(), ignore_index=True)



WFS1=(dataknown.staining == 0).astype(int).sum()
WFS1predicted=(dataUnknown2.staining == 0).astype(int).sum()

Reelin=(dataknown.staining == 1).astype(int).sum()
Reelinpredicted=(dataUnknown2.staining == 1).astype(int).sum()

PV=(dataknown.staining == 2).astype(int).sum()
PVpredicted=(dataUnknown2.staining == 2).astype(int).sum()

pied=[WFS1, WFS1predicted, Reelin, Reelinpredicted, PV, PVpredicted]
my_labels = ['WFS1','WFS1 predicted', 'Reelin', 'Reelin predicted', 'PV', 'PV predicted']
plt.pie(pied,labels=my_labels,colors=['dimgrey','lightgray','steelblue','lightskyblue','darkred','firebrick'],  autopct='%1.1f%%')
plt.title('Parasubiculum Neurons')
plt.axis('equal')
plt.show()


# alldatafinal.to_csv('alldatafinal.csv')

"""
cv = StratifiedKFold(n_splits=5,shuffle=True) #random state default.  no 10 kfold since we have only 7 PV cells!!
model = RandomForestClassifier(n_estimators=100, class_weight=weights, n_jobs=-1)# evaluate model
scoresrocauc=cross_val_score(model, X,y, scoring='roc_auc_ovo', cv=cv, n_jobs=-1)
scoresaccuracy=cross_val_score(model, X,y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean ROC AUC: %.3f' % np.mean(scoresrocauc))
print('Mean accuracy score: %.3f' % np.mean(scoresaccuracy))
# cv.get_n_splits(X, y)



accuracy=[]
balanced_accuracy=[]
precision=[]
recall=[]
f1score=[]

fpr = dict()
fprzero=[]
fprones=[]
fprtwo=[]
tpr = dict()
tprzero=[]
tprones=[]
tprtwo=[]
roc_auc = dict()
roc_auczero=[]
roc_aucones=[]
roc_auctwo=[]

fprmicro=[]
tprmicro=[]
roc_aucmicro=[]

fprmacro=[]
tprmacro=[]
roc_aucmacro=[]
# scores=[]
rocauc=[]

for train_index, test_index in cv.split(X,y):

    print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    # sm = SMOTE(random_state=1)
    # x_train, y_train = sm.fit_resample(x_train,y_train)

    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    
    #probability of confidence let's say 99% sure to put it on class 2 
    #or 34% for 0 4%for 2 and the remaining for 1 . so it goes to 1
    y_predroc=model.predict_proba(x_test) 
    
    # Binarize the output in order to calculate and plot the roc auc scores for each class.
    y_test2 = label_binarize(y_test, classes=[0, 1, 2])
    y_pred2=label_binarize(y_pred, classes=[0,1,2])
    n_classes = y_test2.shape[1]
    y_score=y_pred2
  
    # feature_importances=pd.Series(model.feature_importances_, index=data.columns[0:21])
    # feature_importances=feature_importances.sort_values()
  
    # plt.figure()
    # feature_importances.plot(kind="barh")
    # plt.title(cv)

  
    
    accuracy.append(accuracy_score(y_test,y_pred))
   
    balanced_accuracy.append(balanced_accuracy_score(y_test,y_pred)) #shall I put sample weights?
    
    precision.append(precision_score(y_test, y_pred, average='macro'))
    
    recall.append(recall_score(y_test, y_pred, average='macro'))
    
    f1score.append(f1_score(y_test, y_pred, average='macro'))
   
    rocauc.append(roc_auc_score(y_test, y_predroc, multi_class='ovo',labels=labels,average='macro',max_fpr=1))

    cf=confusion_matrix(y_test, y_pred, normalize='all')
    # plot_confusion_matrix(model,x_test,y_test,normalize='all')
    plot_confusion_matrix(model,x_test,y_test)

    # target_names=['class 0=WFS1', 'class 1=Reelin', 'class 3=PV']

    # report=classification_report(y_test, y_pred, target_names=target_names)

### ROC AUC PLOTS FOR EACH FOLD.############################


    #IMPORTANT NOTE. according to sklearn roc_curve is used for bibary and multilabel classes
    #i use it here to have a look at the sensitivity and specifivity for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if i==0:
            fprzero.append(fpr[i])
            tprzero.append(tpr[i])
            roc_auczero.append(roc_auc[i])
        elif i==1:
              fprones.append(fpr[i])
              tprones.append(tpr[i])
              roc_aucones.append(roc_auc[i])
        else:
             fprtwo.append(fpr[i])
             tprtwo.append(tpr[i])
             roc_auctwo.append(roc_auc[i])
   
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test2.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    fprmicro.append(fpr["micro"])
    tprmicro.append(tpr["micro"])
    roc_aucmicro.append(roc_auc["micro"])
    
    
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    
    #now save it to a list as to have the values for each cross validation cycle! :-)
    fprmacro.append(fpr["macro"])
    tprmacro.append(tpr["macro"])
    roc_aucmacro.append(roc_auc["macro"])
    
    
    
    
# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
        
 
    
    

    
##-------------------------------------------------------------
##threshold analysis
dfWFS1=dataknown.loc[dataknown['staining'] == 0]
dfReelin=dataknown.loc[dataknown['staining'] == 1]
dfPV=dataknown.loc[dataknown['staining'] == 2]  


# stats.ttest_ind(dfWFS1['minISI'], dfReelin['minISI'],equal_var=False)
# stats.ttest_ind(dfWFS1['minISI'], dfPV['minISI'],equal_var=False)
# stats.ttest_ind(dfPV['minISI'], dfReelin['minISI'],equal_var=False)

# stats.ttest_ind(dfWFS1['Min dV/dt'], dfReelin['Min dV/dt'],equal_var=False)
# stats.ttest_ind(dfWFS1['Min dV/dt'], dfPV['Min dV/dt'],equal_var=False)
# stats.ttest_ind(dfPV['Min dV/dt'], dfReelin['Min dV/dt'],equal_var=False)



# stats.ttest_ind(dfWFS1['Halfwidth'], dfReelin['Halfwidth'],equal_var=False)
# stats.ttest_ind(dfWFS1['Halfwidth'], dfPV['Halfwidth'],equal_var=False)
# stats.ttest_ind(dfPV['Halfwidth'], dfReelin['Halfwidth'],equal_var=False)

# stats.ttest_ind(dfWFS1['Max firing freq'], dfReelin['Max firing freq'],equal_var=False)
# stats.ttest_ind(dfWFS1['Max firing freq'], dfPV['Max firing freq'],equal_var=False)
# stats.ttest_ind(dfPV['Max firing freq'], dfReelin['Max firing freq'],equal_var=False)

# stats.ttest_ind(dfWFS1['Membrane capacitance'], dfReelin['Membrane capacitance'],equal_var=False)
# stats.ttest_ind(dfWFS1['Membrane capacitance'], dfPV['Membrane capacitance'],equal_var=False)
# stats.ttest_ind(dfPV['Membrane capacitance'], dfReelin['Membrane capacitance'],equal_var=False)


# HW=sns.boxplot(x="staining",y="Halfwidth",data=data).set_title('known')
# HWu=sns.boxplot(x="staining",y="Halfwidth",data=dataUnknown).set_title('predictions')

# FF=sns.boxplot(x="model",y="totalperformance.columns",data=totalperformance).set_title('known')
# FFu=sns.boxplot(x="staining",y="Max firing freq",data=dataUnknown).set_title('predictions')



# minISI=sns.boxplot(x="staining",y="minISI",data=dataknown).set_title('known')
# minISIu=sns.boxplot(x="staining",y="minISI",data=dataUnknown).set_title('predictions')


# capacitance=sns.boxplot(x="staining",y="Membrane capacitance",data=dataknown).set_title('known')
# capacitanceu=sns.boxplot(x="staining",y="Membrane capacitance",data=dataUnknown).set_title('predictions')


# mindvdt=sns.boxplot(x="staining",y="Min dV/dt",data=dataknown).set_title('known')
# mindvdtu=sns.boxplot(x="staining",y="Min dV/dt",data=dataUnknown).set_title('predictions')

medianReelin=dfReelin.median()
reelin=pd.DataFrame(medianReelin)

reelin=reelin.T
medianPv=dfPV.median()
PV=pd.DataFrame(medianPv)
PV=PV.T

medianWFS1=dfWFS1.median()
WFS1=pd.DataFrame(medianWFS1)
WFS1=WFS1.T

s=pd.concat([WFS1, reelin, PV])
s=s.set_index(s['staining'])
# s=pd.Series([meanReelin,meanPv,meanWFS1], index=['Reelin', 'PV','WFS1'])
lalaa=dataUnknown.loc[['160817_4']]
lalaa1=dataUnknown.loc[['170127_1']]
lalaa2=dataUnknown.loc[['190930_4']]
lalaa3=dataUnknown.loc[['191021_6']]


cells=pd.concat([lalaa,lalaa1,lalaa2,lalaa3])


lalala=pd.concat([s,cells])


""""""
"""



