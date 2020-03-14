'''
Created on Mar 1, 2020

@author: et186010
'''

from article_embedding import read
import pandas as pd

unique_words = set([':', ',', '?', '!', '.', '-', '$', '#', '@', '_', '/', '\\', '\''])

df = pd.read_csv("article_data_sampled.csv", sep="\t")
article_content = df['content'].tolist()

count = 0

for content in article_content:
    tokens = read(content) # replace with Austin's API
    for token in tokens:
        unique_words.add(token)
    
    count = count + 1

word_to_id = {token: idx for idx, token in enumerate(unique_words)}

vfo = open('article_voc_map.properties', "w")
for key in word_to_id:
    strs = str(word_to_id[key]) + "=" + key + "\n"
    vfo.writelines(strs)

vfo.flush()
vfo.close()

print("First pass done")

#print(word_to_id['I']) # save to vocabulary file

def map_text_to_ids(text):
    strs = ""
        
    tokens = read(text) #text.split() # replace with Austin's example
    for token in tokens:
        strs = strs + str(word_to_id[token]) + " "
    
    strs = strs[:-1]
    return strs

fo = open('article_data_word_numeric.csv', "w")

#fo.close()
strs = ""

for content in article_content:
    ids = map_text_to_ids(content)
    ids = ids + "\n"
    fo.writelines(ids)
    
    if count % 1000 == 0:
        print (str(count))
        
    count = count + 1

fo.close()


    
#print (dog)
