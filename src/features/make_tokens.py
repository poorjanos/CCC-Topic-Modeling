import pandas as pd
from pathlib import Path
import spacy
import hu_core_ud_lg as hu
import re


def clean(text):
    '''
    Keeps only word chars and single white space from input text
    '''
    res = re.sub(r'[^\w\s]', '', text)
    res = ' '.join([word.strip() for word in res.split()])
    return res


def tokenize_1gram(text, model, ents = []):
    '''
    Returns lowercase lemma for 
        -non-stop words, 
        -non-numbers, 
        -non-punct 
        and drops
        -lemma of lenght 1
        -entities defined by ents
    '''
    
    doc = model(text)
    res = []
    
    for word in doc:
        if not word.is_stop and not word.is_punct and not word.like_num and len(word.lemma_) > 1 and word.ent_type_ not in ents:
            res.append(word.lemma_.lower())
            
    return res


def remove_single(df, tokens_col):
    '''
    Removes words that only appear once in corpus 
    
    Parameters:
    ----------
    pd_series : pandas Series (df col)
    
    Returns:
    ----------
    Cleaned docs as list of lists

    '''
    all_tokens = sum(df[tokens_col], [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    tokenized_no_single = [[word for word in text if word not in tokens_once] for text in df[tokens_col]]

    return tokenized_no_single
    
    
if __name__ == '__main__':
    # Load hu model
    nlp = hu.load()
    
    # Load corpus
    path_data = Path('../../data')
    df = pd.read_csv(path_data / 'processed' / 'input_data.csv')
    

    
    # Run tokenizer
    df['tokenized_raw'] = df['text'].apply(lambda x: tokenize_1gram(clean(x), model = nlp, ents = ['PER', 'LOC']))
    df['tokenized_raw_cnt'] = df['tokenized_raw'].apply(lambda x: len(x))
    df['tokenized_mults'] = remove_single(df, 'tokenized_raw')
    df['tokenized_mults_cnt'] = df['tokenized_mults'].apply(lambda x: len(x))
    
    # Drop extra stops if available
    try:
        extra_stops = pd.read_csv(path_data / 'processed'/ 'extra_stops.csv')
        df['tokenized_mults_extr'] = df['tokenized_mults'].apply(lambda x: [token for token in x if token not in list(extra_stops['token'])])
        df['tokenized_mults_extr_cnt'] = df['tokenized_mults_extr'].apply(lambda x: len(x))
    except:
        pass
    
    df.to_csv(path_data / 'processed' / 'tokenized1gram_data.csv', index = False)


