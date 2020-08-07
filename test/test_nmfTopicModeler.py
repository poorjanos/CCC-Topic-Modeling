# Test suite for class nmfTopicModeler
from pathlib import Path
import pandas as pd
sys.path.append(str(Path.cwd().parent))
from src.models.nmfTopicModeler import nmfTopicModeler


# Import data
path_to_data = Path.cwd().parent / 'data'
df = pd.read_csv(path_to_data / 'processed' /'tokenized1gram_data.csv',\
                 converters={'tokenized': eval, 'tokenized_mults': eval, 'tokenized_mults_extr': eval})

# Initilaize class instance
myNMF = nmfTopicModeler(n_components = 8, max_iter = 1000)

# Testing fit method
myNMF.fit(df['tokenized_mults_extr'])

# Testing show topics method
myNMF.show_topics(8)

# Testing drop topics method
myNMF.drop_topics(drop_list = [0,4,6])
myNMF.show_topics(8)

# Testing get_topic_frequency method
myNMF.get_topic_frequency(topic_names =\
                          ['pm collection', 'annual', 'arrears sent', 'pm postal', 'pm direct'], threshold = 0.01)

# Testing get_main_topic method
_, main_topic = myNMF.get_main_topic(topic_names = ['pm collection', 'annual', 'arrears sent', 'pm postal', 'pm direct', 'NAN'],\
                                    threshold = 0)
main_topic

# Testing get_topic_patterns method
doc_pattern, topic_patterns = myNMF.get_topic_patterns(topic_names = ['pm collection', 'annual', 'arrears sent', 'pm postal', 'pmdirect'], threshold = 0.015, max_elems = 2)
topic_patterns

# Testing fit_submodels method
doc_sub_patterns, sub_patterns, sub_freqs = myNMF.fit_submodels(df['tokenized_mults_extr'], doc_pattern, topn_topics=5, n_tokens=20, n_topics=5, n_words=8, threshold = 0.02)