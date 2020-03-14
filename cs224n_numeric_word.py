'''
Created on Mar 11, 2020

@author: et186010
'''
import pandas as pd

num_word = {}

vfi = open('article_voc_map.properties', "r")
vfi_props = vfi.readlines()
for vfi_prop in vfi_props:
    row = vfi_prop.split('=')
    str = row[1].split('\n')
    num_word[row[0]] = str[0]

vfi.close()

fo = open('article_data_numeric_word.csv', "w")
count = 0

df = pd.read_csv("article_data_word_numeric.csv", sep=",")
article_content = df['content'].tolist()

for content in article_content:
    tokens = content.split(' ')
    content = ''
    
    for token in tokens:
        content = content + num_word[token] + " "
    
    content = content + "\n"
    fo.writelines(content);
    
    if count % 1000:
        fo.flush()
         
    count = count + 1

fo.close()
