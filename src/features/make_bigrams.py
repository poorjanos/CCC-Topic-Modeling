import pandas as pd
from pathlib import Path
import spacy
import hu_core_ud_lg as hu
import re
import sys
sys.path.append(str(Path.cwd().parent))
from features.make_tokens import clean, tokenize_1gram, remove_single


def remove_stop(model, words):
    '''
    Sets words in model's vocabulary to non-stop
    '''
    for word in words:
        model.vocab[word].is_stop = False


def gen_bigram(doc, extra_stop_list):
    ans = [bigram for bigram in zip(doc[:-1], doc[1:])\
           if bigram[0] not in extra_stop_list and bigram[1] not in extra_stop_list]
    return ans
        
        
# Load hu model
nlp = hu.load()

# Remove default stop words
#remove_stop(nlp, ['nem', 'NEM', 'NEm', 'Nem'])

# Test
#[(i.text, i.is_stop) for i in nlp.vocab if i.text.lower() == 'nem']

# Load corpus
path_data = Path('../../data')
df = pd.read_csv(path_data / 'processed' / 'input_data.csv')
    
# Run tokenizer
# Gen tokens
df['tokenized_raw'] = df['text'].apply(lambda x: tokenize_1gram(clean(x), model = nlp, ents = ['PER', 'LOC']))
df['tokenized_raw_cnt'] = df['tokenized_raw'].apply(lambda x: len(x))

# Gen bigrams from tokens
extra_stops = pd.read_csv(path_data / 'processed'/ 'extra_stops.csv')
extra_stops = list(extra_stops['token'])
df['bigram_raw'] = df['tokenized_raw'].apply(lambda x: gen_bigram(x, extra_stops))
df['bigram_raw_cnt'] = df['bigram_raw'].apply(lambda x: len(x))

# Filter out single bigrams
df['bigram_mults'] = remove_single(df, 'bigram_raw')
df['bigram_mults_cnt'] = df['bigram_mults'].apply(lambda x: len(x))

df.to_csv(path_data / 'processed' / 'tokenized_bigram_data.csv', index = False)
