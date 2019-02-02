#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: Exploration & decisions are justified in explore&report.ipynb
#       This notebook also constitutes the project report
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# see last task

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

import pandas as pd
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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# REFER to explore&report.ipynb

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# REFER to explore&report.ipynb

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from sklearn.tree import DecisionTreeClassifier

features=['to_poi_ratio', 'expenses', 'shared_receipt_with_poi', 'from_poi_to_this_person'] 
# no scaling required
clf = DecisionTreeClassifier(class_weight='balanced',max_depth=4,max_features=4,min_samples_split=4)

dump_classifier_and_data(clf, my_dataset, features_list)

from tester import load_classifier_and_data,test_classifier
### load up student's classifier, dataset, and feature_list
clf, dataset, feature_list = load_classifier_and_data()
### Run testing script
test_classifier(clf, dataset, feature_list,folds=1000)