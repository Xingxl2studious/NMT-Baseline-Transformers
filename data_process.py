import os
import torch
from source.data import batch_tokenize_fn
from functools import partial
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from source.data import create_translation_data
from datasets import load_dataset

source_lang = 'th'
target_lang = 'en'
lang_pair = 'en-th'
pretrained_model = 'facebook/mbart-large-50'
max_length = 128
direction = os.path.join('data', lang_pair)
save_path = os.path.join(direction, f'tokenized_{source_lang}2{target_lang}.pkl')



tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
# model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)


for split in ['train', 'valid', 'test']:
    source_path = os.path.join(direction, f'{lang_pair}.{source_lang}.{split}')
    target_path = os.path.join(direction, f'{lang_pair}.{target_lang}.{split}')
    output_path = os.path.join(direction, f'{lang_pair}.{split}')
    create_translation_data(source_path, target_path, output_path)
    

data_files = {}
for split in ['train', 'valid', 'test']:
    dir_path = os.path.join(direction, f'{lang_pair}.{split}')
    data_files[split] = [dir_path]


dataset_dict = load_dataset(
    'csv',
    delimiter=r'\t',
    column_names=[source_lang, target_lang],
    data_files=data_files
)
print(dataset_dict)

my_batch_tokenize_fn = partial(batch_tokenize_fn, 
                            source_lang=source_lang,
                            target_lang=target_lang,
                            pretrained_tokenizer=tokenizer, 
                            max_source_length=max_length,
                            max_target_length=max_length)
tokenized_dataset_dict = dataset_dict.map(my_batch_tokenize_fn, batched=True)
print(tokenized_dataset_dict)

torch.save(tokenized_dataset_dict, save_path)
print('data process done!')
