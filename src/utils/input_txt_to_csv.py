from pathlib import Path
import os
import io
import pandas as pd


# Get path for input txt files
input_path = Path('../../data/input_txt')
p = input_path.glob('*.txt')
files = [f for f in p if f.is_file()]

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
output_path = Path('../../data/')
df.to_csv(output_path / 'input_data.csv', index = False)


