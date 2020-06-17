import pandas as pd
from pathlib import Path
import spacy
import hu_core_ud_lg as hu
import re


# Load corpus
input_path = Path('../../data')
df = pd.read_csv(input_path / 'input_data.csv')

# Load hu model
nlp = hu.load()


def clean(text):
    '''
    Keeps only word chars and single white space from input text
    '''
    res = re.sub(r'[^\w\s]', '', text)
    res = ' '.join([word.strip() for word in res.split()])
    return res


def tokenize(text, ents = []):
    '''
    Returns lowercase lemma for 
        -non-stop words, 
        -non-numbers, 
        -non-punct 
        and drops
        -lemma of lenght 1
        -entities defined by ents
    '''
    doc = nlp(text)
    res = []
    
    for word in doc:
        if not word.is_stop and not word.is_punct and not word.like_num and len(word.lemma_) > 1 and word.ent_type_ not in ents:
            res.append(word.lemma_.lower())
            
    return res
    
df['tokenized'] = df['text'].apply(lambda x: tokenize(clean(x), ents = ['PER', 'LOC']))
df

