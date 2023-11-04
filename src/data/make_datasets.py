from download_dataset import download_url
import zipfile

url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
output_path = "../../data/raw/filtered.zip"
output_tsv = "../../data/raw/"

download_url(url, output_path)

with zipfile.ZipFile(output_path, "r") as zip_ref:
    zip_ref.extractall(output_tsv)

# remove the zip file
import os

os.remove(output_path)

import pandas as pd

df = pd.read_csv(output_tsv+"filtered.tsv", sep="\t", index_col=[0])

def swap_toxicity(row):
    toxic_second = row['ref_tox'] < row['trn_tox']
    if toxic_second:
        tmp = row['trn_tox']
        row['trn_tox'] = row['ref_tox']
        row['ref_tox'] = tmp
        tmp = row['translation']
        row['translation'] = row['reference']
        row['reference'] = tmp
    return row
    
df = df.apply(swap_toxicity, axis = 1)

df = df.sort_values(by='trn_tox', ascending=False)

output_path = "../../data/interim/swaped.tsv"
df.to_csv(output_path, sep='\t', index=False, header=True) 

MAX_SIZE = 50000
df = df.head(MAX_SIZE)

output_path = "../../data/interim/processed.tsv"
df.to_csv(output_path, sep='\t', index=False, header=True) 


