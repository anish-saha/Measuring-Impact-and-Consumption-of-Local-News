# Script to merge and preprocess both award and non-award 
# article data csv files @author: Anish Saha
import pandas as pd
import sys, csv, re, time
from datetime import datetime

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def cleanhtml(x):
    x = str(x)
    cleantext = re.sub(re.compile('<.*?>'), '', x)
    return cleantext

def process_utc(x):
    s = str(x)
    if '(' in s:
        yyyy = re.match('tm_year=(\d{4})', s).group(1)
        mm = re.match('tm_mon=(\d{1,2})', s).group(1)
        if int(mm) < 10: mm = '0' + mm
        dd = re.match('tm_mday=(\d{1,2})', s).group(1)
        if int(dd) < 10: dd = '0' + dd
        s = yyyy + '-' + mm + '-' + dd
        s = datetime.strptime(s, '%Y-%m-%d')
        return time.mktime(s.timetuple())
    else:
        if s == 'nan':
            return 'nan'
        if '.0' in s:
            return x
        else:
            try:
                s = re.sub('\\s.*', '', s)
                s = datetime.strptime(s, '%Y-%m-%d')
                return time.mktime(s.timetuple())
            except:
                print(s)
                return 'nan'

# Retrieve article data
award_df = pd.read_csv('award_article_data.csv', sep='\t', engine='python')
award_df['award'] = [1] * len(award_df)
no_award_df = pd.read_csv('article_data_small.csv', sep='\t', engine='python')
no_award_df['award'] = [0] * len(no_award_df)
no_award_df['datetime-publish']=pd.to_datetime(no_award_df['datetime-publish'],
                                               errors='coerce', yearfirst=True,
                                               utc=True)

# Merge and preprocess datframes
df = pd.concat([award_df, no_award_df], ignore_index=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['content (NEED TO PREPROCESS!)'] = df['content (NEED TO PREPROCESS!)'].apply(cleanhtml)
df['datetime-publish'] = df['datetime-publish'].apply(process_utc)
df = df.drop(['date'], axis=1)

df.to_csv('article_data.csv', sep='\t')
print("Merge successful.")

