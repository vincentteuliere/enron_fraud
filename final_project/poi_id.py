#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: Exploration & decisions are justified in explore.ipynb
#       This notebook also constitutes the project report
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Udacity function modified to:
#  return scores 
#  random seed removed

def my_test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return accuracy, precision, recall, f1, f2
    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")



### Task 1: Select what features you'll use.

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
        
# In this code, fature list is built automatically based on features selection 
# (see below Task 3)
# Some variations can be expected depending on random generators seeds
# but it doesn't change drastically the performance of the final classifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

import pandas as pd
import numpy as np
# labels of features
EMAILS=['to_messages', 
        'email_address', 
        'from_poi_to_this_person', 
        'from_messages', 
        'from_this_person_to_poi', 
        'shared_receipt_with_poi']

STOCK_VALUE=['exercised_stock_options',
             'restricted_stock',
             'restricted_stock_deferred', 
             'total_stock_value']
                    
PAYMENTS=['salary', 
          'bonus',
          'long_term_incentive',
          'deferred_income',
          'deferral_payments', 
          'loan_advances',
          'other', 
          'expenses', 
          'director_fees',
          'total_payments']
FEATURES=PAYMENTS+STOCK_VALUE+EMAILS
# to pandas dataframe
df=pd.DataFrame(data_dict).transpose()

# Replacing NaN per numpy NaN
def replace_nan(s):
    if s=='NaN':
        return(np.nan)
    else:
        return(s)      
    
df=df.applymap(replace_nan)

### Task 2: Remove outliers

# THESE DECISIONS ARE FULLY JUSTIFIED in following sections of explore.ipynb

#2.Explore, clean, improve and check dataset
#•Explore dataset
#•Clean dataset
#•Improve dataset
#•Outliers detection

# filling financial features NaN with zeros
df[PAYMENTS]=df[PAYMENTS].fillna(0)
df[STOCK_VALUE]=df[STOCK_VALUE].fillna(0)

# filling EMAILS NaN features with the median of population
poi=df[df.poi].index.values
not_poi=df[~df.poi].index.values
df.loc[poi,EMAILS]=df[df.poi][EMAILS].fillna(df[df.poi][EMAILS].median())
df.loc[not_poi,EMAILS]=df[~df.poi][EMAILS].fillna(df[~df.poi][EMAILS].median())

# Removing Eugene whose features are fully filled with NaN values
df=df.drop('LOCKHART EUGENE E',axis=0)
# and the travel agency
df=df.drop('THE TRAVEL AGENCY IN THE PARK',axis=0)

# Correct values
df.loc['BELFER ROBERT',PAYMENTS]=[0,0,0,-102500,0,0,0,3285,102500,3285]
df.loc['BELFER ROBERT',STOCK_VALUE]=[0,44093,-44093,0]
df.loc['BHATNAGAR SANJAY',PAYMENTS]=[0,0,0,0,0,0,0,137864,0,137864]
df.loc['BHATNAGAR SANJAY',STOCK_VALUE]=[15456290,2604490,-2604490,15456290]
df.loc['GLISAN JR BEN F','shared_receipt_with_poi']=873

# drop loan advances
df.total_payments=df.total_payments-df.loan_advances
df=df.drop('loan_advances',axis=1)
PAYMENTS.remove('loan_advances')
FEATURES.remove('loan_advances')

# drop director_fees
df.total_payments=df.total_payments-df.director_fees
df=df.drop('director_fees',axis=1)
PAYMENTS.remove('director_fees')
FEATURES.remove('director_fees')

# drop restricted_stock_deferred
df.total_stock_value = df.total_stock_value - df.restricted_stock_deferred
df=df.drop('restricted_stock_deferred',axis=1)
STOCK_VALUE.remove('restricted_stock_deferred')
FEATURES.remove('restricted_stock_deferred')

# drop email address
df=df.drop('email_address',axis=1)
EMAILS.remove('email_address')
FEATURES.remove('email_address')

# drop TOTAL
df=df.drop('TOTAL',axis=0)

### Task 3: Create new feature(s)


'''

COMMENT from reviewer:

Given the limitations of our dataset (low number of data points, low proportion 
of POIs), we may accept the engineered features from a practical standpoint. 
However, using some features like from_poi_to_this_person, from_this_person_to_poi, 
shared_receipt_with_poi to engineer new features can potentially create a data leakage. 
This can hamper the model s ability to generalize on unseen data and can give the 
false effect that the model performs really well. The features mentioned contain
 information that is used to create the feature, while also containing information 
 about the target (POI/Non-POI).
 
 MY ANSWER & questions:
 
 This comment sounds a bit meaningless.
 
 I concur that from_poi_to_this_person, from_this_person_to_poi, 
 shared_receipt_with_po features are biased as they contain information about 
 the target. But the point is that all these features are part of the original 
 dataset. Here, we only built features based on existing ones, we are 
 neither introducing bias or information. The bias is already existing in the 
 dataset because of these features.
 
 Are we supposed to fully remove these features? It would make sense but why 
 including them in the dataset? Was this a trap? Could you clarify?    

 If we need to do so, just turn on the boolean below: all biased 
 email features will be removed from the dataset. Resulting F1-score drops by 
 30 to 40%.

 
 '''

REMOVE_BIASED_EMAILS_FEATURES=False
 
if REMOVE_BIASED_EMAILS_FEATURES:
     for feature in ['from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi']:
         df.drop(feature,axis=1)
         EMAILS.remove(feature)
         FEATURES.remove(feature)
else:     
    # emails engineering features
    df['email_ratio_poi']=(df.from_poi_to_this_person+df.from_this_person_to_poi)/(df.from_messages+df.to_messages)
    df['from_poi_ratio']=df.from_poi_to_this_person/df.to_messages
    df['to_poi_ratio']=df.from_this_person_to_poi/df.from_messages
    df['shared_poi_ratio']=df.shared_receipt_with_poi/df.to_messages
    EMAILS_RATIO = ['email_ratio_poi','from_poi_ratio','to_poi_ratio','shared_poi_ratio']
    # update list of features
    FEATURES += EMAILS_RATIO

# financial engineering features

# custom normalization to avoid division per zero
def normalize_to_max(row):
    total=row[-1]
    max=total
    if max==0.0:
        max = row.max()
    if max==0.0:
        max = 1
    result = row/max
    result[-1] = total
    return result

STOCK_VALUE_RATIO=[s+'_ratio' for s in STOCK_VALUE]
PAYMENTS_RATIO=[s+'_ratio' for s in PAYMENTS]

# Normalize stock features per total (or max if total is null)
df[STOCK_VALUE_RATIO]=df[STOCK_VALUE].apply(normalize_to_max,axis=1)
# Normalize payments -same method-
df[PAYMENTS_RATIO]=df[PAYMENTS].apply(normalize_to_max,axis=1)
df=df.drop(STOCK_VALUE_RATIO[-1],axis=1)
df=df.drop(PAYMENTS_RATIO[-1],axis=1)
# global ratios
df['ratio_payment']=df[PAYMENTS[-1]]/(df[STOCK_VALUE[-1]]+df[PAYMENTS[-1]])
df['ratio_stocks']=df[STOCK_VALUE[-1]]/(df[STOCK_VALUE[-1]]+df[PAYMENTS[-1]])
# Two employees have null divider
df=df.fillna(0)
# Updating features lists
STOCK_VALUE_RATIO=[s+'_ratio' for s in STOCK_VALUE[:-1]]
PAYMENTS_RATIO=[s+'_ratio' for s in PAYMENTS[:-1]]
STOCK_VALUE_RATIO += ['ratio_stocks']
PAYMENTS_RATIO += ['ratio_payment']
FEATURES += PAYMENTS_RATIO + STOCK_VALUE_RATIO

### Store to my_dataset for easy export below.
my_dataset = df.transpose().to_dict()

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

'''
 As requested by previous reviewer, we develop here, the method
 used for feature selection, tuning and evaluation for the best classifier
 DTC. 
 
 All details are provided for severals different features selection methods,
 and classifiers (DTC, SVC, KNN, Logistic Regression, Naive Bayes) in the 
 jupyter notebook explore.ipynb. Refer to following sections:
3.Features ranking
•Preliminary exploration
•Filter methods
•RFE methods
•Embedded methods        
4.Classifiers evaluation
•Validation of classifiers versus features selection methods

 Basically, the jupiter code just explore the combinatory. But the core 
 principle is detailed below (e.g main loop)
'''

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Constants for step 1. feature selection (ranking)
SELECT_FEATURES=True # if False step is skipped
N_CV_FEATURE_SELECT=1000 # Gridsearch cross-validation per run
RATIO_CV_TRAIN__FEATURE_SELECT=0.7 #train/test ratio of cv

# Constants for step 2. tune the number of features
TUNE_FEATURES=True # if False this step is skipped
MAX_FEATURES=15    # above features are not examined

# Constants for step 3 tuning classifier parameters
N_LOOP_TUNING=10 # to illiustrate variance & variability between runs
N_CV_TUNING=100 # Gridsearch cross-validation per run
RATIO_CV_TRAIN_TUNING=0.7 #train/test ratio of cv
MAXIMIZE_RECALL = False # if true will minimize depth of tree


# Dataset to numpy arrays
Y = df.poi.values        
X = df[FEATURES].values

# 1. Recursive feature selection with cross-validation using classifier DTC
# with defaut parameters apart class_weight is set to 'balanced', because 
# classes are highly imbalanced 

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This step is applied to many others features classification methods in the
# jupiter notebook: filter (pearson, MIC, F-test), RFECV as below, random forest
# and random logistic Regression
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('1. Selecting features')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
if SELECT_FEATURES:
    df_feature_score=pd.DataFrame(index=FEATURES) 
    clf = DecisionTreeClassifier(class_weight='balanced')
    rfecv = RFECV(
                    estimator=clf,
                    cv=StratifiedShuffleSplit(
                                                y=Y,
                                                n_iter=N_CV_FEATURE_SELECT,
                                                train_size=RATIO_CV_TRAIN__FEATURE_SELECT
                                            ),
                    scoring='f1'
                 )
    rfecv = rfecv.fit(X,Y)
    df_feature_score['RFE-DTC ranking']=rfecv.ranking_
    print('RANKING of FEATURES based on RFE-DTC and',N_CV_FEATURE_SELECT,'StratifiedShuffleSplit\n')
    print(df_feature_score.sort_values(by='RFE-DTC ranking',ascending=True))

    
# 2. Evaluate performance of the classifier depending on the number of features
# ranked above

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This step is applied to all classifiers & feature selection methods in the 
# jupiter notebook
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('2. TUNING number of features')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
df_result=pd.DataFrame()
if TUNE_FEATURES:
    
    if SELECT_FEATURES:    
        FEATURES_ranked=df_feature_score.sort_values('RFE-DTC ranking',ascending=True).index.values
    else:
        FEATURES_ranked = FEATURES
        MAX_FEATURE = len(FEATURES)
        
    fmax=0
    nfeat=0
    clf = DecisionTreeClassifier(class_weight='balanced')
    for n in range(1,MAX_FEATURES+1):
        # train & cross-validate classifier
        
        accuracy, precision, recall, f1, f2 = my_test_classifier(clf, 
                                                              my_dataset, 
                                                              ['poi']+list(FEATURES_ranked[0:n]))
    
        if f1>fmax:
            fmax=f1
            nfeat=n
            print('n={0:d},f1={1:.2f}<=feature added={2:s}'.format(n,f1,FEATURES_ranked[n-1]))
    
        # record performance 
        df_result.loc[n,'accuracy']=accuracy
        df_result.loc[n,'precision']=precision
        df_result.loc[n,'recall']=recall
        df_result.loc[n,'f1']=f1
        df_result.loc[n,'f2']=f2
        
    print('=> Best f1 score (',fmax,') obtained with following',nfeat,'features:')
    FEATURES_selected = FEATURES_ranked[0:nfeat]
    print(FEATURES_selected)
else:
    
    if SELECT_FEATURES:
        FEATURES_selected=FEATURES_ranked
    else:
        FEATURES_selected=FEATURES


# 3. Tune classifier parameters 
#    To minimize overfit, idea is to minimize 
#    max_depth,  max_feature & min_samples_split
#    Features are selected during step 1. & 2

print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('3. TUNING DTC classifier')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
    
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This step is also applied to SVC & KNN in the jupiter notebook
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

df_result = pd.DataFrame(columns=['max_depth','max_features','min_samples_split','score'])


tuned_parameters=[{
        'max_depth':(range(2,7,1)),
        'max_features':(range(2,nfeat,1)),
        'min_samples_split':(range(2,8,2))}]

X_tune=df[FEATURES_selected].values
    
for i in range(0,N_LOOP_TUNING):
    grid = GridSearchCV(
                    clf, 
                    tuned_parameters, 
                    cv=StratifiedShuffleSplit(
                            y=Y,
                            n_iter=N_CV_TUNING,
                            train_size=RATIO_CV_TRAIN_TUNING),
                    scoring='f1')    
    grid = grid.fit(X_tune,Y)

    best_clf=grid.best_estimator_

    tmp = grid.best_params_
    tmp['score']=grid.best_score_
    df_result.loc[i,:]=tmp

print(df_result)
print('\n')
    

if MAXIMIZE_RECALL:
    # select values which minimize overfitting, e.g minimum depth
    # maximize recall
    max_depth =  int(df_result.min()['max_depth'])
else:
    # maximize depth and precision
    max_depth =  int(df_result.max()['max_depth'])

# select two others parameters according max score
max_score = df_result[df_result.max_depth==max_depth]['score'].max()
max_features =  int(df_result[df_result.score==max_score].max_features)
min_samples_split =  int(df_result[df_result.score==max_score].min_samples_split)

print('Parameters selected to mitigate overfit:')
print('max_depth=',max_depth)
print('max_features=',max_features)
print('min_samples_split=',min_samples_split)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# See 3. above 

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=max_depth,
                             max_features=max_features,
                             min_samples_split=min_samples_split)

features_list=['poi']+list(FEATURES_selected)


dump_classifier_and_data(clf, my_dataset, features_list)

from tester import load_classifier_and_data,test_classifier
### load up student's classifier, dataset, and feature_list
clf, dataset, feature_list = load_classifier_and_data()
### Run testing script
test_classifier(clf, dataset, features_list)