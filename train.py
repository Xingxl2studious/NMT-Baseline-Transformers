import os
import torch
 
import random
import argparse
import numpy as np
import sentencepiece as spm

import sklearn
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer
)
from source.data import Seq2SeqDataCollator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_lang", type=str
    )
    parser.add_argument(
        "--target_lang", type=str
    )
    parser.add_argument(
        "--lang_pair", type=str
    )
    parser.add_argument(
        "--saved_dir",
        type=str,
        default='checkpoint'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default='facebook/mbart-large-50'
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_sentence_length", 
        type=int, 
        default=128)
    parser.add_argument(
        "--freeze_decoder",
        type=int,
        default=0
    )
    args = parser.parse_args()
    return args

def setup_seed(seed):
    print('seed is: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('setup seed')


print("------------")
args = parse_args()
print(f'source lang: {args.source_lang}')
print(f'target lang: {args.target_lang}')

print(f'saved direction: {args.saved_dir}')
print(f'batch_size: {args.batch_size}')
src_lang = args.source_lang
tgt_lang = args.target_lang
lang_pair = args.lang_pair
max_length = args.max_sentence_length
device = torch.device('cuda:0')
setup_seed(int(args.seed))
print(args.pretrained_model)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

# load model
translation_model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)

# load data
direction = os.path.join('data', lang_pair)
print(os.path.join(direction, f'tokenized_{src_lang}2{tgt_lang}.pkl'))
tokenized_dataset_dict = torch.load(os.path.join(direction, f'tokenized_{src_lang}2{tgt_lang}.pkl'))


if args.freeze_decoder == 1:
    modules = [translation_model.model.decoder, translation_model.lm_head, translation_model.model.shared]
else:
    modules = [translation_model.lm_head, translation_model.model.shared]
for module in modules:
    for name, param in module.named_parameters():
        param.requires_grad = False

checkpoint_path = f'checkpoint/{lang_pair}/{args.saved_dir}/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    evaluation_strategy="steps",
    learning_rate=args.lr,
    logging_dir=checkpoint_path+'log', #
    logging_strategy="steps", #
    logging_steps=100, #
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=200,
    seed=int(args.seed),
    load_best_model_at_end=True,
    predict_with_generate=True,
    remove_unused_columns=True,
    fp16=True,
    gradient_accumulation_steps=2,
    eval_steps=500,
    warmup_steps=100,
    dataloader_pin_memory=False,
)

data_collator = Seq2SeqDataCollator(max_length, tokenizer.pad_token_id, tokenizer.pad_token_id)
callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

trainer = Seq2SeqTrainer(
    model=translation_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["valid"],
    callbacks=callbacks,
)
print('training')
trainer_output = trainer.train()
