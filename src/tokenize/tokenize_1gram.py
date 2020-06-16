import pandas as pd
from pathlib import Path

input_path = Path('../../data')
df = pd.read_csv(input_path / 'input_data.csv')
df.head()

