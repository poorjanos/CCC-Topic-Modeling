from pathlib import Path
import os
import io
import pandas as pd


def gather_txt_to_csv(input_path_to_txt, output_path_to_csv):
    '''
    Gather txt files in folder to one csv keeping txt filename col
    '''
    print(f'Starting to process txt files at {input_path_to_txt}')
    
    # Get path for input txt files
    input_paths = Path(input_path_to_txt).glob('*.txt')
    files = [path for path in input_paths if path.is_file()]

    # Collect file names and content
    file_name_and_content = {}
    for file in files:
        with io.open(file, mode="r") as target_file:
             file_name_and_content[file.stem] = target_file.read()

    # Add to pandas.DataFrame            
    df = pd.DataFrame.from_dict(file_name_and_content, orient='index')
    df.reset_index(inplace = True)
    df.columns = ['file_name', 'text']
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    # Save
    df.to_csv(Path(output_path_to_csv) / 'input_data.csv', index = False)
    
    print(f'Finished processing. Result available at: {output_path_to_csv}')

    
if __name__ == '__main__':
    gather_txt_to_csv('../../data/input_txt', '../../data/')
