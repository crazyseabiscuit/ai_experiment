from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import argparse

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

parser = argparse.ArgumentParser(description='tokenizer train process')
parser.add_argument('-f', '--file', type=str, default='example',help='Train File path')
parser.add_argument('-o','--output', type=str, default='tokenizer_output', help='output tokenizer path')

args = parser.parse_args()
file_path = args.file
output = args.output
if(file_path == 'example'):
  raw_datasets = load_dataset("code_search_net", "python")
  print(raw_datasets)
  print(raw_datasets["train"][123456]["whole_func_string"])
  training_corpus = get_training_corpus()
else:
  print(f'file_path:{file_path}')

base_vocab = list(bytes_to_unicode().values())
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

example = "click_button naivigation_page click_submit"
tokens = tokenizer.tokenize(example)
print(tokens)
tokenizer.save_pretrained(output)



def get_training_corpus():
    return (
        df[i : i + 1000]["content"]
        for i in range(0, len(df, 1000))
)
df = pd.read_csv('data/clickstream-enwiki-2023-10.tsv',sep='\t', names=['prev', 'curr', 'type','occur'],\
    encoding='utf-8',on_bad_lines = 'warn').head(20000)
df['content'] = df['prev'] + ' ' + df['curr'] + ' ' + df['type']

