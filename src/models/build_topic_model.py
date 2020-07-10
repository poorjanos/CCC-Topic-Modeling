# Helper functions for topic modeling
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim import models
from gensim.models import CoherenceModel

def lda_compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    LDA: Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=2,
            update_every=1,
            passes=10,
            alpha='auto',
            per_word_topics=True)
        
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def show_topics(a, num_top_words, vocab):
    '''
    Show topics with n highest weighted words
    '''
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]


def drop_topics(w, h, drop_list):
    '''
    Drop topics from NMF model
    '''
    keeps = [i for i in range(w.shape[1]) if i not in drop_list]
    return w[:, keeps], h[keeps, :]


def sparse_argsort(arr):
    '''
    Argsort for 1D array ignoring zeros
    '''
    indices = np.nonzero(arr)[0]
    return indices[np.argsort(arr[indices])]

def get_topic_patterns(dtm, topic_names, threshold=0):
    '''
    Extract topic patterns from doc-topic matrix returned by NMF
    Filters  non-relevant topics by threshold
    
    Parameters:
    ----------
    dtm : doc-topic matrix from NMF
    threshold: min weight for topic to be picked up
    lut: look-up table for topic names
    
    Returns:
    -------
    Pandas df with topic patterns and frequency metrics
    '''
    
    # Set limit for topic weigth and zero out topics below
    mask = dtm >= threshold
    masked_dtm = np.where(mask, dtm, 0)
    
    # Extract topic patterns
    topic_patterns = [sparse_argsort(i) for i in masked_dtm]
    topic_patterns = [tuple(np.array(topic_names)[i]) for i in topic_patterns]
    
    # Add to df and compute frequency metrics
    topic_patterns_df = pd.DataFrame()
    topic_patterns_df['pattern'] = topic_patterns
    
    topic_freq = topic_patterns_df.groupby(['pattern']).size().reset_index(name = 'count')\
            .sort_values(by=['count'], ascending = False).reset_index(drop = True)
    topic_freq['pct'] = topic_freq['count']/sum(topic_freq['count'])
    topic_freq['pct roll'] = topic_freq['count'].cumsum()/sum(topic_freq['count'])
    
    return topic_freq