'''
Created on Mar 1, 2020

'''
"""
Script to translate articles to word embeddings

"""

#import pandas as pd
import re
def read(article):
        output = []
        article = article.lower()
        #sentences = article.split(".  ")
        sentences = re.split('[?.!]'+"  ", article)
        for i in range(len(sentences)):
                sentence= sentences[i]
                words = re.findall(r"[\w']+|[.,!?;]", sentence)
                for word in words:
                    output.append(word)
        return output
#articles = articles.str.lower()
