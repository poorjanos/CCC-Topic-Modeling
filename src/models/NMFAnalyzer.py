import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from pathlib import Path

class NMFAlanyzer(object):
    def __init__(self, doc_topic_matrix, topic_term_matrix, vocab):
        self.doc_topic_matrix = doc_topic_matrix
        self.topic_term_matrix = topic_term_matrix
        self.vocab = vocab
    
    
    @classmethod
    def from_corpus(cls, corpus, topic_number = 1, max_iter = 100, random_state = 0):
        tfidf = TfidfVectorizer(analyzer='word', tokenizer=lambda doc: doc,\
                                preprocessor = lambda doc: doc, token_pattern=None)
        
        doc_term_matrix = tfidf.fit_transform(corpus)
        
        mdl = NMF(init='random', n_components=topic_number, max_iter=max_iter, random_state=random_state)
        mdl.fit(doc_term_matrix)
        
        doc_topic_matrix = mdl.fit_transform(doc_term_matrix)
        topic_term_matrix = mdl.components_
        vocab = np.array(tfidf.get_feature_names())
        
        return cls(doc_topic_matrix, topic_term_matrix, vocab)
    
    
    def show_topics(self, num_top_words):
        '''
        Show topics with n highest weighted words
        '''
        top_words = lambda t: [self.vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
        topic_words = ([top_words(t) for t in self.topic_term_matrix])
        return [' '.join(t) for t in topic_words]
    

# Testing
path_to_data = Path.cwd().parent.parent / 'data'
df = pd.read_csv(path_to_data / 'processed' /'tokenized1gram_data.csv',\
                 converters={'tokenized': eval, 'tokenized_mults': eval, 'tokenized_mults_extr': eval})
df.head()

myNMF = NMFAlanyzer.from_corpus(df['tokenized_mults_extr'], topic_number = 8)
myNMF.show_topics(3)
