'''
Created on Mar 12, 2020

@author: et186010
'''

from article_embedding import read
import re

from deeppavlov import configs, build_model

ner_model = build_model("/Users/cs224n/deeppavlov/lib/python3.6/site-packages/deeppavlov/configs/ner/ner_conll2003_bert.json", download=True)

ner_result = ner_model("/Users/cs224n/deeppavlov_predict/article_data_50_word.txt")

fo = open('/Users/cs224n/deeppavlov_predict/article_data_50_word_ner_result.txt', "w")
for result in ner_result:
    fo.writelines(result)
    fo.writelines('\n')
fo.close()

fi = open('/Users/cs224n/deeppavlov_predict/article_data_50_word_ner_result.txt', "r")
count = 0

lines = fi.readlines()
source = []
tag = []
entities = []

line_count = 0
for line in lines:
    print (line_count)
    words = re.findall(r"[\w\-.!?',\]]+", line)
    tagging = False
    first = False
    article_src = []
    article_tag = []
    article_entities = []
    for index in range(len(words)):
        if tagging == False:
            if words[index] != '],':
                article_src.append(words[index])
            elif words[index] == '],':
                tagging = True
                first = True
            else:
                article_src.append(words[index])
                tagging = True
                first = True
        else:
            if words[index] == ']]':
                continue;
            
            article_tag.append(words[index])

    #if len(article_src) != len(article_tag):
    #    print(len(article_src))
    #    print(len(article_tag))
    
    line_count = line_count + 1
    strs = ''
    start = False
    for index in range(len(article_src)):
        if article_tag[index] == 'B-PER':
            start = True
            strs = strs + article_src[index]
        elif article_tag[index] == 'I-PER':
            strs = strs + " " + article_src[index]
        elif article_tag[index] == 'B-LOC':
            start = True
            strs = strs + article_src[index]
        elif article_tag[index] == 'I-LOC':
            strs = strs + " " + article_src[index]
        elif article_tag[index] == ',':
            continue
            
        elif start == True:
            article_entities.append(strs)
            start = False
            strs = ''
        
    entities.append(article_entities)
            
#print (entities)
#print(tag)
fi.close()

fo = open('/Users/cs224n/deeppavlov_predict/article_data_50_word_ner_result_extract.txt', "w")

for entity in entities:
    fo.writelines(entity)
    fo.writelines('\n')

fo.close()



