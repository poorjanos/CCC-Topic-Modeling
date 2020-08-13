import pandas as pd
import io
from pathlib import Path


def build_sentiment_dict(input_path_to_txt, output_path_to_csv, output_filename):
    '''
    Clean and unify sentiment dicts
    '''
    print(f'Starting to process txt files at {input_path_to_txt}')
    
    # Get path for input txt files
    input_paths = input_path.glob('*.txt')
    files = [path for path in input_paths if path.is_file()]

    # Collect file names and content
    file_name_and_content = {}
    for file in files:
        with io.open(file, mode="r", encoding="utf-8") as target_file:
            content = target_file.readlines()
            file_name_and_content[file.stem] = [word.strip().lower() for word in content]    
    
    # Join sentiment dicts and drop duplicates
    sentiment = {}
    sentiment['neg'] = ' '.join(sorted(set(file_name_and_content['negative_words_hu'] + file_name_and_content['PrecoNeg'])))
    sentiment['pos'] = ' '.join(sorted(set(file_name_and_content['positive_words_hu'] + file_name_and_content['PrecoPos'])))
    
    # Add to df
    df = pd.DataFrame.from_dict(sentiment, orient='index')
    df = df.reset_index().rename(columns={'index': 'sentiment', 0: 'words'})

    # Save
    output_filename = output_filename + '.csv'
    df.to_csv(Path(output_path_to_csv) / output_filename, index = False)
    
    print(f'Finished processing. Result available at: {output_path_to_csv}')


if __name__ == '__main__':
    # Read in raw data
    input_path = Path.cwd().parent.parent / 'data' / 'input_sentiment'
    output_path = Path.cwd().parent.parent / 'data' / 'processed'
    build_sentiment_dict(input_path, output_path, output_filename='sentiment_dict')



