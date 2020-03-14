# Script to run 4 classifiers on dataset with all extracted features.
# Also measures performance metrics and feature importances of the various
# classification models used: Logistic Regression, Random Forest, Gradient
# Boosting, and Multi-Layer Perceptron (MLP) neural network classifiers.
# @Author: Anish Saha | Created: 03-10-2020
import pandas as pd
import numpy as np
import sklearn.metrics as skm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from category_encoders.target_encoder import TargetEncoder

# Data Processing
df = pd.read_csv('article_data_sampled_features.csv', sep='\t')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# BERT-NER Pretrained Model Entity Extraction Feature
entities = []
f = open('article_data_50_word_ner_result_extract.txt', 'r')
for val in f:
    if val == '\n': entities.append('none')
    else: entities.append(val.replace('\n', ''))
df['bert_ner_entity'] = entities

# Word2Vec Embeddings k-Means Cluster Labels Feature (k=5)
word2vec_clusters = pd.read_csv('word2vec_cluster_labels.csv', sep='\t')
df['word2vec_cluster_label'] = word2vec_clusters['word2vec_cluster_label']

# Target Variable preprocessing
df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
temp = df.pop('award')
df['award'] = temp

# Training-Validation-Test Set Split
np.random.seed(seed=42)
neg_idx = df.loc[df['award'] == 0].index.tolist()
pos_idx = df.loc[df['award'] == 1].index.tolist()
train_neg_idx = np.random.choice(neg_idx, 6200, replace=False)
train_pos_idx = np.random.choice(pos_idx, 1800, replace=False)
train_idx = np.append(train_neg_idx, train_pos_idx)

dev_set = df.loc[df.index.isin(train_idx)]
test_set = df.loc[~df.index.isin(train_idx)]

X, y = dev_set.iloc[:,:-1], dev_set.iloc[:,-1]
X_test, y_test = test_set.iloc[:,:-1], test_set.iloc[:,-1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

columns_to_encode = ['bert_ner_entity', 'section', \
                     'word2vec_cluster_label', 'author']
for col in columns_to_encode:
    encoder = TargetEncoder()
    X_train[col] = encoder.fit_transform(X_train[col].astype(str), y=y_train)
    X_val[col] = encoder.transform(X_val[col].astype(str))  
    X_test[col] = encoder.transform(X_test[col].astype(str))

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

# Gradient Boosting Classifier
print("\n============================\nGRADIENT BOOSTING CLASSIFIER\n============================")

model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
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

# Multi-Layer Perceptron Neural Network Classifier
print("\n================================================\nMULTI-LAYER PERCEPTRON NEURAL NETWORK CLASSIFIER\n================================================")

model = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', \
                      solver='adam', alpha=0.0001, batch_size=200, \
                      learning_rate_init=0.001, max_iter=10000, \
                      random_state=42).fit(X_train, y_train)
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

# Classification Task Complete!
print("\nSuccess! Evaluation metrics displayed above.\n")

