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
tokenizer = RegexpTokenizer(r'\w+')
G=nx.DiGraph()
# create English stop words list
en_stop = get_stop_words('en')
docset = []
prev='p'
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
					else:
						G.add_node(ids)
						G.add_edge(ids,prev,weight=5)
					prev=ids
					#print ids
	#print child[0][0].text
	#print child[0].attrib
					docset.append(child[2].text)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
#tree = ET.parse('1916247.0000.xml')
'''tree = ET.parse('1916247.0000.xml')
root = tree.getroot()
for child in root:
	for post in child.findall('user'):
		ids=post.get('id')
		G.add_node(ids)
		print ids
	#print child[0][0].text
	#print child[0].attrib
	docset.append(child[2].text)
#print docset	
'''
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 
s1="i will fuck you"
s2="this is a great house"
# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
doc_aadil=[s1,s2]
# list for tokenized documents in loop
texts = []

# loop through document list
'''
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)
'''
for i in docset:
    
    # clean and tokenize document string
    #print i
    raw = i.lower
    tokens = tokenizer.tokenize(i)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)
	
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
tmp = ldamodel.show_topics(num_topics=10, num_words=5, log=False, formatted=True)
print tmp
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
index = similarities.MatrixSimilarity(tfidf[corpus])
sims = index[corpus_tfidf]
print sims
nx.draw(G)
plt.show()
a = nx.hits(G)
print a
#print ldamodel.show_topics(2)
#print ldamodel.num_topics
