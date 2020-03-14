# Script to retrieve synthetic features outlined in Human Designed Features.docx
# @Authot: Anish Saha 
import pandas as pd
import numpy as np
import re

df = pd.read_csv('article_data_sampled.csv', sep='\t')

# Feature Extraction
def compute_index(x, keywords):
    total = 0
    for word in keywords:
        total += x.count(word)
    return total

def extract_accountability_index(x):
    keywords = ['abuse', 'corruption', 'fraud', 'misconduct', \
                'mismanagement', 'neglect', 'risk', 'stealing', \
                'investigation', 'evidence', 'follow-up']
    return compute_index(x, keywords)

def extract_official_records(x):
    keywords = ['according to school district records', 'audits', \
                'according to court records', 'documents found', \
                'according to police records', 'documents state', \
                'FOIA', ' Freedom of Information Act', 'records', \
                'court case', 'official document', 'documents show']
    return compute_index(x, keywords)

def extract_anecdotal_data_index(x):
    quote_words = re.findall(r'"(.*?)"', x)
    total = 0
    for i in quote_words:
        total += len(i.split())
    return total

def extract_quantitative_data_index(x):
    keywords = ['statistic', 'numerical', 'quantitative', 'percent', \
                'fraction', 'graph', 'chart', 'percent', '%']
    total = compute_index(x, keywords)
    for i in x.split():
        if i.isnumeric(): total += 1
        elif len(i.split('.')) == 2:
            if i.split('.')[0].isnumeric() and i.split('.')[1].isnumeric():
                total += 1
    return total

def extract_investigation_index(x):
    if re.match(r'\d{1,2}.*month.*investigation', x):
        return 1
    elif re.match(r'months.*long.*investigation', x):
        return 1
    elif re.match(r'year.*long.*investigation', x):
        return 1
    elif re.match(r'years.*long.*investigation', x):
        return 1
    else: 
        return 0

def extract_breaking_news(x):
    keywords = ['massacre', 'survivors', 'deadliest', 'FBI', 'fbi',
                'breaking news', 'latest news']
    return compute_index(x, keywords)

def process_shorten(x):
    try: return x[:50]
    except: return 'none'

print('Dataframe loaded.')

# Text Analysis Synthetic Feature Extraction
df['word_count'] = df['content'].apply(lambda x: len(x.split()))
print('word_count done.')
df['accountability_index'] = df['content'].apply(extract_accountability_index)
print('accountability_index done.')
df['logistical_record_index'] = df['content'].apply(extract_official_records)
print('logistical_record_index done.')
df['anecdotal_data_index'] = df['content'].apply(extract_anecdotal_data_index)
print('anecdotal_data_index done.')
df['quant_data_index'] = df['content'].apply(extract_quantitative_data_index)
print('quant_data_index done.')
df['investigation_index'] = df['content'].apply(extract_investigation_index)
print('investigation_index done.')
df['breaking_news_index'] = df['content'].apply(extract_breaking_news)
print('breaking_news_index done.')

'''
DEBUG
'''
print(df.columns)
print(df['word_count'].value_counts())
print(df['accountability_index'].value_counts())
print(df['logistical_record_index'].value_counts())
print(df['anecdotal_data_index'].value_counts())
print(df['quant_data_index'].value_counts())
print(df['investigation_index'].value_counts())
print(df['breaking_news_index'].value_counts())

#print(df.columns)
#print(df.describe())

df = df.drop(['content'], axis=1)
df['author'] = df['author'].apply(process_shorten)
df['section'] = df['section'].apply(process_shorten)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.to_csv('article_data_sampled_features.csv', sep='\t')
print("\nFeature extraction complete.\n")

