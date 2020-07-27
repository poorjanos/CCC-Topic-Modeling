import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter


class nmfTopicModeler(NMF):
    '''
    Extends sklearn.decomposition.NMF by adding methods for
    topic model interpretation.
    Parameters:
    -----------
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    '''
    def __init__(self, init = 'random', n_components=1,\
                 max_iter=100, random_state=0, *args, **kwargs):
        super().__init__(init=init, n_components=n_components,\
                         max_iter=max_iter, random_state=random_state, *args, **kwargs)
          
    def fit(self, tokenized_corpus, vectorizer = 'tfidf'):
        '''
        Fit NMF topic model
        Parameters:
        -----------
        tokenized_corpus: tokenized corpus as a list of documents
        vectorizer: method to use in building document-term matrix
        '''
        # Set method to build document-term matrix
        if vectorizer == 'tfidf':
            self.vectors = TfidfVectorizer(analyzer='word', tokenizer=lambda doc: doc,\
                                preprocessor = lambda doc: doc, token_pattern=None)
        elif vectorizer == 'count':
            self.vectors = CountVectorizer(analyzer='word', tokenizer=lambda doc: doc,\
                                preprocessor = lambda doc: doc, token_pattern=None)
        
        # Build document-term matrix
        self.doc_term_matrix = self.vectors.fit_transform(tokenized_corpus)
        
        # Fit nmf model
        self.mdl = super().fit(self.doc_term_matrix)
        
        # Model outputs
        self.doc_topic_matrix = self.mdl.fit_transform(self.doc_term_matrix)
        self.topic_term_matrix = self.mdl.components_
        self.vocab = np.array(self.vectors.get_feature_names())
    
    
    def show_topics(self, num_top_words):
        '''
        Show topics with n highest weighted words
        '''
        top_words = lambda t: [self.vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
        topic_words = ([top_words(t) for t in self.topic_term_matrix])
        return [' '.join(t) for t in topic_words]
    
    
    def drop_topics(self, drop_list):
        '''
        Drop topics from NMF model
        '''
        keeps = [i for i in range(self.doc_topic_matrix.shape[1]) if i not in drop_list]
        self.doc_topic_matrix =  self.doc_topic_matrix[:, keeps]
        self.topic_term_matrix =  self.topic_term_matrix[keeps, :]
        print(f'Dropped topics {",".join(map(str, drop_list))}')
    
    
    @staticmethod
    def sparse_argsort(arr):
        '''
        Argsort for 1D array ignoring zeros
        Returns empty array if all elements are 0
        '''
        indices = np.nonzero(arr)[0]
        return indices[np.argsort(arr[indices])]
    
    
    def get_topic_frequency(self, topic_names, threshold=0):
        '''
        Extract how frequently a topic occurs in corpus from doc-topic matrix returned by NMF
        Filters non-relevant topics by threshold
        Parameters:
        ----------
        threshold: min weight for topic to be picked up
        topic_names: look-up table for topic names
        Returns:
        -------
        Pandas df with topic frequency metrics
        '''

        # Set limit for topic weigth and zero out topics below
        mask = self.doc_topic_matrix >= threshold
        masked_dtm = np.where(mask, self.doc_topic_matrix, 0)

        # Extract topic frequency
        topic_patterns = [nmfTopicModeler.sparse_argsort(i) for i in masked_dtm]
        topic_patterns = [np.array(topic_names)[i] for i in topic_patterns]
        topic_patterns_flat = [item for sublist in topic_patterns for item in sublist]
        counts = pd.DataFrame.from_dict(Counter(topic_patterns_flat),\
                                        orient='index', columns = ['count'])
        counts['pct'] = counts['count']/sum(counts['count'])
        counts.sort_values(by=['count'], ascending = False, inplace = True)

        return counts

    
    def get_main_topic(self, topic_names, threshold=0):
        '''
        Extract the dominant topic for each doc in doc-topic matrix returned by NMF
        Filters non-relevant topics by threshold
        Parameters:
        ----------
        threshold: min weight for topic to be picked up
        topic_names: look-up table for topic names
        Returns:
        -------
        main_topic: Pandas df with main topic for each doc
        main_stas: Pandas df with aggregated measures of main topic distribution
        '''
        mask = self.doc_topic_matrix >= threshold
        masked_dtm = np.where(mask, self.doc_topic_matrix, 0)

        main_topic = [nmfTopicModeler.sparse_argsort(i) for i in masked_dtm]
        main_topic = [row[len(row)-1] if len(row) > 0 else len(topic_names)-1 for row in main_topic]
        main_topic = [topic_names[i] for i in main_topic]
        main_topic = pd.DataFrame(main_topic, columns = ['topic'])


        main_stats = main_topic.groupby(['topic']).size()
        main_stats = pd.DataFrame(main_stats, columns = ['count']).sort_values(by=['count'], ascending = False)
        main_stats['pct'] = main_stats['count']/sum(main_stats['count'])
        main_stats['pct roll'] = main_stats['count'].cumsum()/sum(main_stats['count'])

        return main_topic, main_stats
    
    
    def get_topic_patterns(self, topic_names, threshold=0):
        '''
        Extract topic patterns from doc-topic matrix returned by NMF
        Filters  non-relevant topics by threshold
        Parameters:
        ----------
        threshold: min weight for topic to be picked up
        topic_names: look-up table for topic names
        Returns:
        -------
        topic_patterns: Pandas df with topic pattern for each doc
        patterns_stats: Pandas df with aggregated measures of topic pattern distribution
        '''

        # Set limit for topic weigth and zero out topics below
        mask = self.doc_topic_matrix >= threshold
        masked_dtm = np.where(mask, self.doc_topic_matrix, 0)

        # Extract topic patterns
        patterns = [nmfTopicModeler.sparse_argsort(i) for i in masked_dtm]
        patterns = [tuple(np.sort(np.array(topic_names)[i])) for i in patterns]

        # Add to df and compute frequency metrics
        topic_patterns = pd.DataFrame()
        topic_patterns['pattern'] = patterns

        patterns_stats = topic_patterns.groupby(['pattern']).size().reset_index(name = 'count')\
                .sort_values(by=['count'], ascending = False).reset_index(drop = True)
        patterns_stats['pct'] = patterns_stats['count']/sum(patterns_stats['count'])
        patterns_stats['pct roll'] = patterns_stats['count'].cumsum()/sum(patterns_stats['count'])
        patterns_stats.set_index('pattern', inplace = True)

        return topic_patterns, patterns_stats


