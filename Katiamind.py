# For Katia ^-^.

import string
import sklearn
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from urllib import request
from bs4 import BeautifulSoup as bs
from os import path
from glob import glob
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster import KMeansClusterer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer

from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import cluster
import matplotlib.pyplot as plt
from pandas import DataFrame
from networkx.drawing.nx_agraph import graphviz_layout


#######################################


rsrcmgr = PDFResourceManager()
retstr = StringIO()
codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    
fp = open(r'F:\Books\Business\finance\riskmanager.pdf','rb')
interpreter = PDFPageInterpreter(rsrcmgr, device)
password = ""
maxpages = 764
caching = True
pagenos=set()
for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
    interpreter.process_page(page)
text = retstr.getvalue()

fp.close()
device.close()
retstr.close()
    
   



########## tokenize word and sentences  
senttokens=sent_tokenize(text)
   
tokenization=word_tokenize(text)
  

#defining all the stop words 
stop_words=set(nltk.corpus.stopwords.words('english'))
newStopWords = ['the','\'','.','of','a','to','The','(','"','C',',',')',':','Figure','!','A','i','one','M.','also','B','L','E','O','N',']','[',';','...']
morestopwords = stop_words.union(newStopWords)
newstopwords1=['surfaces','Appl','In','P','J','P.','shown']
newstopwords2=morestopwords.union(newstopwords1)
filtered_sentence = [w for w in tokenization if not w in newstopwords2] 
filtered_sentence = [] 
  
for w in tokenization: 
    if w not in newstopwords2: 
        filtered_sentence.append(w) 
# tokenization process here     
print(filtered_sentence) 
print(tokenization)
print(senttokens) 
fdist = FreqDist(filtered_sentence)
print(fdist)
fdist.plot(50,cumulative=False)
plt.show()
    
wordcloud = WordCloud(width=900,height=500, max_words=50,relative_scaling=1,normalize_plurals=False).generate_from_frequencies(fdist)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show() 
#   
  
#Clustering the data 
vect = TfidfVectorizer(min_df=1)
tfidf = vect.fit_transform(senttokens)
vectors=(tfidf * tfidf.T).A
print(vectors)

initial = [cluster.vq.kmeans(vectors,i) for i in range(1,10)]
pyplot.plot([var for (cent,var) in initial])
pyplot.show()

cent, var = initial[3]
#use vq() to get as assignment for each obs.
assignment,cdist = cluster.vq.vq(vectors,cent)
pyplot.scatter(vectors[:,0], vectors[:,1],c=assignment)
pyplot.show()
   
    

    





