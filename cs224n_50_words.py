'''
Created on Mar 1, 2020

@author: et186010
'''
from article_embedding import read
import pandas as pd

df = pd.read_csv("article_data_sampled.csv", sep="\t")
article_content = df['content'].tolist()

fo = open('article_data_50_word.csv', 'w')
strs = ""
count = 0

for content in article_content:
    tokens = read(content)
    strs = ""
    count = 0
    for token in tokens:
        count = count + 1
        strs = strs + token + " "
        if count > 48:
            break

    strs = strs + "\n"
    fo.writelines(strs)
    if count % 1000 == 0:
        fo.flush()
        print(str(count))
        count = count + 1  

fo.close()
