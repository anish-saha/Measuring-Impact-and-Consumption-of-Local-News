# -*- coding: UTF-8 -*-
import sys
import numpy as np
import random
import scipy as sp
import pandas as pd
from sklearn.decomposition import TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)

#articles_toy = []
#articles_toy.append("It takes five days on average for people to start showing the symptoms of coronavirus, scientists have confirmed.\n The Covid-19 disease, which can cause a fever, cough and breathing problems, is spreading around the world and has already affected more than 114,000 people.\n The US team analysed known cases from China and other countries to understand more about the disease.\n Most people who develop symptoms do so on or around day five.\n Anyone who is symptom-free by day 12 is unlikely to get symptoms, but they may still be infectious carriers., U.S. stocks plunged more than 7.5\% in the worst day on Wall Street since the financial crisis, as a full-blown oil price war rattled financial markets already on edge over the spreading coronavirus. Treasury yields plummeted, crude sank 20\% and credit markets��buckled.\n The S&P 500 sank the most since December 2008, the Dow Jones Industrial Average tumbled 2,000 points and small caps lost more than 9\% as investors fled risk assets with virus cases surging and the Trump administration so far unwilling to step in to soften the expected economic blow.")
#articles_toy.append("Business stocks economics, the stock market crashes and jumps and stocks go down. When will the stocks stop dropping?")
#articles_toy.append("The virus is killing many people.  Doctors are worried and a quarnatine is in effect.  Many symptoms have been described as deadily.  The disease is now considered a pandemic, and people are advised to stay indoors.")
print("initial read")
df = pd.read_csv("article_data_sampled.csv", sep="\t")
max_art_ind = 10000
print("Convert to list")
articles_long = df['content'].tolist()[:max_art_ind]
print("shorten")
articles_toy = [elem[:100] for elem in articles_long]# shorten each article to a fixed size (computational reasons)
print(len(articles_toy[0]))
def read_corpus(articles):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    #files = articles#reuters.fileids(category)
    #print(files)
    return [[START_TOKEN] + [w.lower() for w in list(f.split())] + [END_TOKEN] for f in articles]

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    for doc in corpus:
        for word in doc:
            corpus_words.append(word)
            
    corpus_words = sorted(set(corpus_words))
    num_corpus_words = len(corpus_words)

    # ------------------

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    
    # ------------------
    # Write your implementation here.
    M = np.zeros((num_words, num_words))
    #Map
    i = 0
    for word in words:
        word2Ind[word] = i
        i = i+1
    
    #Fill co-occurence matrix
    for i in range(len(corpus)): #for each doc
        doc = corpus[i]
        for j in range(len(doc)):# for each word
            for k in range(len(doc)): #for each word (word X word)
                if(abs(j-k) <= window_size): #within window
                    word1 = doc[j]
                    word2 = doc[k]
                    if(word1 != word2):
                        index1 = word2Ind[word1]
                        index2 = word2Ind[word2]
                        M [index1,index2] = M [index1,index2] +1
        

    # ------------------

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
        # ------------------
        # Write your implementation here.
    svd = TruncatedSVD(n_components = k, n_iter = n_iters,)
    M_reduced=svd.fit_transform(M)
        # ------------------

    print("Done.")
    return M_reduced

from collections import Counter 
def top5_words(text, k):
    counts = Counter(text)
    return [elem for elem, _ in counts.most_common(k)]

print("reading")
toy_corpus = read_corpus(articles_toy)
print("flattening")
flatten_list = [j for sub in toy_corpus for j in sub] 
print(flatten_list)

#M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(toy_corpus)
#print("Co_occurence calculated")
#M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
#print("Reduced...")
#print(M_reduced_co_occurrence)

# Rescale (normalize) the rows to make them each of unit-length
#M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
#M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

#words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
words_unique_all = set(flatten_list)
print(len(words_unique_all))
#print(flatten_list)
# most_common() produces k frequently encountered 
# input values and their respective counts. 
words_unique_most_occur = top5_words(flatten_list, 20)

####plot_embeddings(M_normalized, word2Ind_co_occurrence, words_unique_most_occur)

#Get reduced coords
M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(toy_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)#k=len(words_unique_all)-1)

#print(word2Ind_co_occurrence)
print(len(M_reduced_co_occurrence[0]))
#print(M_reduced_co_occurrence)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting
print(len(M_normalized[0]))
# This is a vocab X vocab matrix of co_occurences


#Over all words, do 3-means clustering with euclidean distance.
from sklearn.cluster import KMeans 
clusters = 12
kmeans = KMeans(n_clusters = clusters) 
kmeans.fit(M_normalized) 
print(kmeans.labels_)
print(len(kmeans.labels_))
#This is a mapping of words to the nearest cluster.

#For each article, take the rounded average label as its "topic"
scores_over_articles = []
for article in toy_corpus:
    length = len(article)
    score = 0
    for word in article:
        index = word2Ind_co_occurrence[word]
        label = kmeans.labels_[index]
        score = score + label
    average_score = score/length
    scores_over_articles.append(average_score)

print(len(scores_over_articles))    
print(scores_over_articles)
scores_over_articles = np.around(scores_over_articles)
print("Rounded...")
print(scores_over_articles)
print(set(scores_over_articles))
df = pd.DataFrame(data={'word2vec_cluster_label': scores_over_articles})
df.to_csv('word2vec_cluster_labels.csv', sep='\t')
print("\nWord2Vec cluster analysis complete.\n")

