# Helper functions for LDA modeling
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