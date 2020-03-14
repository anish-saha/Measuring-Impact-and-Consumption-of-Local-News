# Script to run baseline classifier models and analyze their performance metric
# Fatures used: author, datetime-publish, word-count
# Models used: Logistic Regression, Random Forest
# @author: Anish Saha | Created: 02-24-2020
import pandas as pd
import numpy as np
import sklearn.metrics as skm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('article_data_sampled.csv', sep='\t')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['author'] = LabelEncoder().fit_transform(df['author'].astype(str))
df['word-count'] = df['content'].apply(lambda x: len(x.split()))
df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
df.drop(['section', 'content'], axis=1, inplace=True)
temp = df.pop('award')
df['award'] = temp

np.random.seed(seed=42)
neg_idx = df.loc[df['award'] == 0].index.tolist()
pos_idx = df.loc[df['award'] == 1].index.tolist()
train_neg_idx = np.random.choice(neg_idx, 6200, replace=False)
train_pos_idx = np.random.choice(pos_idx, 1800, replace=False)
train_idx = np.append(train_neg_idx, train_pos_idx)

dev_set = df.loc[df.index.isin(train_idx)]
test_set = df.loc[~df.index.isin(train_idx)]

'''
DEBUG

print(len(dev_set.loc[dev_set.award == 0]))
print(len(dev_set.loc[dev_set.award == 1]))
print(len(test_set.loc[test_set.award == 0]))
print(len(test_set.loc[test_set.award == 1]))

'''

X, y = dev_set.iloc[:,:-1], dev_set.iloc[:,-1]
X_test, y_test = test_set.iloc[:,:-1], test_set.iloc[:,-1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Classifier
print("\n==============================\nLOGISTIC REGRESSION CLASSIFIER\n==============================")

model = LogisticRegressionCV(cv=10, random_state=42).fit(X_train, y_train)
pred = model.predict(X_val)
print(classification_report(y_val, pred))
print("Validation Set Accuracy: " + str(skm.accuracy_score(y_val, pred)))
print("Validation Set F1 Score: " + str(skm.f1_score(y_val, pred)))
print("Validation Set Precision: " + str(skm.precision_score(y_val, pred)))
print("Validation Set Recall: " + str(skm.recall_score(y_val, pred)))
print("Validation Set ROC-AUC: " + str(skm.roc_auc_score(y_val, pred)))
pred = model.predict(X_test)
print("\n")
print(classification_report(y_test, pred))
print("Test Set Accuracy: " + str(skm.accuracy_score(y_test, pred)))
print("Test Set F1 Score: " + str(skm.f1_score(y_test, pred)))
print("Test Set Precision: " + str(skm.precision_score(y_test, pred)))
print("Test Set Recall: " + str(skm.recall_score(y_test, pred)))
print("Test Set ROC-AUC: " + str(skm.roc_auc_score(y_test, pred)))

# Random Forest Classifier
print("\n========================\nRANDOM FOREST CLASSIFIER\n========================")

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
pred = model.predict(X_val)
print(classification_report(y_val, pred))
print("Validation Set Accuracy: " + str(skm.accuracy_score(y_val, pred)))
print("Validation Set F1 Score: " + str(skm.f1_score(y_val, pred)))
print("Validation Set Precision: " + str(skm.precision_score(y_val, pred)))
print("Validation Set Recall: " + str(skm.recall_score(y_val, pred)))
print("Validation Set ROC-AUC: " + str(skm.roc_auc_score(y_val, pred)))
pred = model.predict(X_test)
print("\n")
print(classification_report(y_test, pred))
print("Test Set Accuracy: " + str(skm.accuracy_score(y_test, pred)))
print("Test Set F1 Score: " + str(skm.f1_score(y_test, pred)))
print("Test Set Precision: " + str(skm.precision_score(y_test, pred)))
print("Test Set Recall: " + str(skm.recall_score(y_test, pred)))
print("Test Set ROC-AUC: " + str(skm.roc_auc_score(y_test, pred)))

print("\nFeature Importances:")
for f in zip(X_train.columns.tolist(), model.feature_importances_): print(f)

print("\nSuccess! Baseline evaluation metrics displayed above.\n")
