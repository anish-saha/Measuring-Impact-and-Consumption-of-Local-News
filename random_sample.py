# Short script to randomly sample and extract 10,000 rows from dataframe
# author: Anish Saha
import pandas as pd

df = pd.read_csv('article_data.csv', sep='\t')
df['content'] = df["headlines"] + " " + df['content (NEED TO PREPROCESS!)']
df.drop(['content (NEED TO PREPROCESS!)', 'headlines', 'newspaper'], axis=1, inplace=True) # drop unnecessary columnns and columns that cause bias
df = df.dropna(subset=['content'])
df = df.loc[df['content'].apply(lambda x: not isinstance(x,(float,bool,int)))]
award_df = df.loc[df['award'] == 1].sample(n=2000, random_state=42)
no_award_df = df.loc[(df['award'] == 0)].sample(n=8000, random_state=42)
res = pd.concat([award_df, no_award_df], ignore_index=True)
res = res.loc[:, ~res.columns.str.contains('^Unnamed')]
res.to_csv('article_data_sampled.csv', sep='\t')
print('Sampling complete.')

