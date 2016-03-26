# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models,similarities
import gensim
from collections import defaultdict
from pprint import pprint
from sqlalchemy.engine import create_engine
import json
from xml.dom import minidom
import os
import networkx as nx
import matplotlib.pyplot as plt
import graphlab as gl
import pandas as pd
import pyLDAvis
import pyLDAvis.graphlab
import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer  

#loading stop words file
def loadStopWords(stopWordFile):
    stopWords = []
    for line in open(stopWordFile):
        for word in line.split( ): #in case more than one per line
            stopWords.append(word)
    return stopWords

# remove common words and tokenize
def remove_stopwords(documents):
    stoplist = loadStopWords('englishstop.txt')
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    return texts
	  
#return a JSON representaion of LDA allocation	  
def get_LDAasJSON(texts):
    # Make dictionary
    dictionary = corpora.Dictionary(texts)
    #dictionary.save('test.dict') # store the dictionary, for future reference

    #Create and save corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    #corpora.MmCorpus.serialize('test.mm', corpus) # store to disk, for later use

    #Run LDA
    model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20, update_every=0, passes=20)
    #Save Model
    #model.save('ldamodel.m')

    tmp = model.show_topics(num_topics=20, num_words=5, log=False, formatted=False)

    #print tmp
    json_tm = json.dumps(tmp)

    return json_tm
    
#return TFIDF values for words    
def get_TFIDF(texts):
	vectorizer = TfidfVectorizer(min_df=1)
	X = vectorizer.fit_transform(texts)
	idf = vectorizer.idf_
	return dict(zip(vectorizer.get_feature_names(), idf))
	
G=nx.DiGraph()
# create English stop words list
en_stop = get_stop_words('en')
docset = []
prev='p'
nodelist = []
index=0
pos=nx.spring_layout(G) 
for fn in os.listdir('.'):
	if os.path.isfile(fn):
        	if fn.endswith(".xml"):
        		tree = ET.parse(fn)
        		print fn
			root = tree.getroot()
			for child in root:
				for post in child.findall('user'):
					ids=post.get('id')
					if(prev=='p'):
						G.add_node(ids)
						nodelist.append(ids)
						index+=1
					else:
						G.add_node(ids)
						G.add_edge(ids,prev,weight=5)
						nodelist.append(ids)
						index+=1
					prev=ids
					docset.append(child[2].text)
					
cyberbullyingdocs = remove_stopwords(docset)
dictionary = corpora.Dictionary(cyberbullyingdocs)
tfidf_corpus = get_TFIDF(docset)
print "\n\nTFIDF VALUES:\n\n" 
print tfidf_corpus
topic_model_cyberbullying_json = get_LDAasJSON(cyberbullyingdocs)
print "\n\nLDA Allocation\n\n\n"
print topic_model_cyberbullying_json
nx.draw(G)
a = nx.hits(G)
plt.show()


