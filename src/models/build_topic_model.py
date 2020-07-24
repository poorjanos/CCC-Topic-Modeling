# Helper functions for topic modeling
import numpy as np
import pandas as pd
from collections import Counter
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
    Returns empty array if all elements are 0
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
    topic_names: look-up table for topic names
    
    Returns:
    -------
    topic_patterns: Pandas df with topic pattern for each doc
    patterns_stats: Pandas df with aggregated measures of topic pattern distribution
    '''
    
    # Set limit for topic weigth and zero out topics below
    mask = dtm >= threshold
    masked_dtm = np.where(mask, dtm, 0)
    
    # Extract topic patterns
    patterns = [sparse_argsort(i) for i in masked_dtm]
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


def get_topic_frequency(dtm, topic_names, threshold=0):
    '''
    Extract how frequently a topic occurs in corpus from doc-topic matrix returned by NMF
    Filters non-relevant topics by threshold
    
    Parameters:
    ----------
    dtm : doc-topic matrix from NMF
    threshold: min weight for topic to be picked up
    topic_names: look-up table for topic names
    
    Returns:
    -------
    Pandas df with topic frequency metrics
    '''
    
    # Set limit for topic weigth and zero out topics below
    mask = dtm >= threshold
    masked_dtm = np.where(mask, dtm, 0)
    
    # Extract topic frequency
    topic_patterns = [sparse_argsort(i) for i in masked_dtm]
    topic_patterns = [np.array(topic_names)[i] for i in topic_patterns]
    topic_patterns_flat = [item for sublist in topic_patterns for item in sublist]
    counts = pd.DataFrame.from_dict(Counter(topic_patterns_flat), orient='index', columns = ['count'])
    counts['pct'] = counts['count']/sum(counts['count'])
    counts.sort_values(by=['count'], ascending = False, inplace = True)
    
    return counts



def get_main_topic(dtm, topic_names, threshold=0):
    '''
    Extract the dominant topic for each doc in doc-topic matrix returned by NMF
    Filters non-relevant topics by threshold
    
    Parameters:
    ----------
    dtm : doc-topic matrix from NMF
    threshold: min weight for topic to be picked up
    topic_names: look-up table for topic names
    
    Returns:
    -------
    main_topic: Pandas df with main topic for each doc
    main_stas: Pandas df with aggregated measures of main topic distribution
    '''
    mask = dtm >= threshold
    masked_dtm = np.where(mask, dtm, 0)
    
    main_topic = [sparse_argsort(i) for i in masked_dtm]
    main_topic = [row[len(row)-1] if len(row) > 0 else len(topic_names)-1 for row in main_topic]
    main_topic = [topic_names[i] for i in main_topic]
    main_topic = pd.DataFrame(main_topic, columns = ['topic'])

    
    main_stats = main_topic.groupby(['topic']).size()
    main_stats = pd.DataFrame(main_stats, columns = ['count']).sort_values(by=['count'], ascending = False)
    main_stats['pct'] = main_stats['count']/sum(main_stats['count'])
    main_stats['pct roll'] = main_stats['count'].cumsum()/sum(main_stats['count'])
    
    return main_topic, main_stats