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
    
    
    def get_topic_patterns(self, topic_names, threshold=0, max_elems=None):
        '''
        Extract topic patterns from doc-topic matrix returned by NMF
        Filters  non-relevant topics by threshold
        Parameters:
        ----------
        threshold: min weight for topic to be picked up
        topic_names: look-up table for topic names
        max_elems: limit the number of topics in pattern
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
        if max_elems:
            patterns = [i[-max_elems:] for i in patterns]
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
    
    
    @staticmethod
    def fit_submodels(corpus, topics, topn_topics=None, n_tokens=None, n_topics=5, n_words=8, threshold=0.01):
        '''
        Fit NMF model for subsets of corpus defined by main topics or topic patterns. Returns main topics
        for each submodel
        Parameters:
        ----------
        corpus: tokenized corpus as a list of documents (shape: n_docs x len_corpus)
        topics: main topics or topic patterns for each doc (shape: n_docs x len_corpus)
        topn_topics: consider only most frequent main topics or patterns
        n_tokens: num of tokens to use in submodels
        n_topics: num of topics to extract by submodels
        n_words: num of highest weighted words to show in topics names
        threshold: mmin weight for topic to be picked up in submodels
        Returns:
        -------
        doc_topic: pandas df with main topic/topic pattern and main topics from submodel for each doc
        sub_stats: pandas df with aggregate metrics and 2-level index:
        level 0: main topic or topic pattern
        level 1: main topics from submodel
        '''
        # Get first n tokens
        tokens_c = pd.DataFrame(corpus)
        topic_c = topics.copy()
        tokens_c.columns = ['tokens']
        topic_c.columns = ['topic']
        doc_topic = tokens_c.join(topic_c)
        
        # Filter for topn topics
        if topn_topics:
            doc_topic = doc_topic[doc_topic.topic.isin(doc_topic.groupby('topic').size().sort_values(ascending = False).nlargest(topn_topics).index)]
        
        # Get subtokens
        if n_tokens:
            doc_topic['tokens_sub'] = doc_topic['tokens'].apply(lambda doc: [(k, v)[1] for (k, v) in enumerate(doc) if k < n_tokens])
        else:
            doc_topic['tokens_sub'] = doc_topic['tokens']

        # Fit models for main topics
        submodels_stats = {}
        submodel_docs = {}
        for topic in doc_topic['topic'].unique():
            sub = doc_topic.loc[doc_topic['topic'] == topic, :]
            topic_mdl_sub = nmfTopicModeler(n_components = n_topics, max_iter = 1000)
            topic_mdl_sub.fit(sub['tokens_sub'])
            
            doc_sub, sub_mains = topic_mdl_sub.get_main_topic(topic_names = topic_mdl_sub.show_topics(n_words), threshold = threshold)
            doc_sub.set_index(sub.index, inplace=True)

            submodels_stats[topic] = sub_mains
            submodel_docs[topic] = doc_sub
            
        submodels_stats = pd.concat(submodels_stats.values(), keys=[str(i) for i in submodels_stats.keys()])
        submodel_docs = pd.concat(submodel_docs.values(), keys=[str(i) for i in submodel_docs.keys()])
        return submodel_docs, submodels_stats