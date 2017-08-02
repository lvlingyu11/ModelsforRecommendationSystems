from BaseModel import BaseModel
import sys





import logging
import numpy as np
import nltk
import string
from numpy import genfromtxt
import urllib2
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
from sklearn.metrics.pairwise import linear_kernel


logger = logging.getLogger(__name__)  # __name__ is the module's name

'''TF-IDF is used to measure the similarity between two text documents. It is usually used when documents
such as news, publications. or web pages are to be recommended.
The text documents are converted to tf-idf vectors, cosine similarity are then used to quantify the similarity
of two vectors (documents)'''

class TFRS(BaseModel):
    def __init__(self):
        super(TFRS, self).__init__()
        self.token_dict = {}
        self.topsimilar = {}  # Store the calculated tf-idf for each item
        self.values = None
        self.key = None
        self.indice_docid = None

    def train(self, datasource):
        '''Train the recommendation model.
        First model the short-term interests with the nearest neighbor alg. Create TF-IDF matrix for each paper,
        which means, each paper is converted to its tf-idf representation.
        TF-IDF could look at different num. of word phrases, such as uni-(one), bi-(two), tri-(three) grams. Here,
        we consider each word(one).'''

        # import the dataset
        filename = datasource
        # data = genfromtxt(filename, delimiter=',', dtype = None) # genfromtext does not work here because the title could include comma
        data = pandas.read_csv(filename, header = None, encoding="latin-1").as_matrix()
        print "total number of papers: ", len(data)
        # data has 4 columns: document_id, title, clean_title, and url
        for d in data:
            url = d[3]
            if url == "\N" or url is None or url.isspace():
                continue
            request = urllib2.Request(url)
            try:
                response = urllib2.urlopen(request)
                pdfdata = response.read()
                response.close()
                lowers = pdfdata.lower()
                # remove the punctuation
                no_punctuation = lowers.translate(None, string.punctuation)
                self.token_dict[d[0]] = no_punctuation
                print "document ", d[0]
            except urllib2.URLError as e:
                print e.reason
            except urllib2.HTTPError as e:
                print e.reason

        # create the tf-idf matrix for each paper
        # stemmed data is the clean text, we use them for claculating the tf-idf
        self.values = [] # using tfidf.transform(self.token_dict.values()) will lost any link to the documents ids
        self.key = {}
        self.indice_docid = {}
        index = 0
        for k, v in self.token_dict.items():
            self.values.append(v)
            self.key[k] = index
            self.indice_docid[index] = k
            index += 1
        tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english', decode_error='ignore')
        tfs = tfidf.fit_transform(self.values)   # document-term matrix

        # the results are in a matrix, they are a collection of tf-idf numbers
        # Till now, each paper is converted to its tf-idf representation, and they will be stored in user model

        '''Score prediction for a new paper: calculate the cosine similarity between papers, sort the similarity
        score and then suggest the top n items.'''
        # Calculate the similarity of items, use cosine similarity
        cosine_similarities = linear_kernel(tfs, tfs)
        # For each item, store the top 5 most similar items
        for d in data:
            # get the row index of the corresponding document id
            rid = self.get_rowid(d[0])
            if rid is None:
                continue
            similar_indices = cosine_similarities[rid].argsort()[-5:]  # argsort returns the indices of the sorted array
            # get the document id based on the indices
            similar_ids = self.get_docid(similar_indices)
            # remove the first item, which is the item itself.
            self.topsimilar[d[0]] = similar_ids[1:]

    def tokenize(self, textdata):
        tokens = nltk.word_tokenize(textdata)
        # stemming the filtered data, using the Porter Stemmer in NLTK
        stemmer = PorterStemmer()
        stems = self.stem_token(tokens, stemmer)
        return stems

    def stem_token(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def get_rowid(self, docid):
        if docid not in self.key.keys():
            return None
        return self.key[docid]

    def get_docid(self, similar_indices):
        docids = []
        for i in similar_indices:
            docids.append(self.indice_docid[i])
        return docids

    def predict(self, docid, recnum):
        # Predict the similar items of one item. Just need to retrive the similar items and their score from the storage
        # recnum is the number of items recommended, since we only store the top 5 similar items, so assume recnum<=5
        return self.topsimilar[docid][0:recnum]


        




