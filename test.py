import os
import argparse
import torch
from functools import partial
from source.eval import evaluate_parallel
from transformers import AutoTokenizer
import evaluate


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
        "--pretrained_model",
        type=str,
        default='facebook/mbart-large-50'
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_sentence_length", 
        type=int, 
        default=128)
    parser.add_argument(
        "--device_num",
        type=int,
        default=0
    )
    args = parser.parse_args()
    return args


print("Testing------------")
args = parse_args()
print(f'source lang: {args.source_lang}')
print(f'target lang: {args.target_lang}')

print(f'saved direction: {args.saved_dir}')
print(f'batch_size: {args.batch_size}')
src_lang = args.source_lang
tgt_lang = args.target_lang
lang_pair = args.lang_pair
max_length = args.max_sentence_length
device_num = args.device_num

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

# load data
direction = os.path.join('data', lang_pair)
print(os.path.join(direction, f'tokenized_{src_lang}2{tgt_lang}.pkl'))
tokenized_dataset_dict = torch.load(os.path.join(direction, f'tokenized_{src_lang}2{tgt_lang}2.pkl'))


my_evaluate_parallel = partial(evaluate_parallel,
                            pretrained_model=args.pretrained_model,
                            tokenized_dataset_dict=tokenized_dataset_dict,
                            max_sentence_length=max_length,
                            device_num=device_num,
                            batch_size=args.batch_size
                            )

checkpoint_path = os.path.join('checkpoint', lang_pair, args.saved_dir)
if 'result.txt' in os.listdir(checkpoint_path):
    print('already have result.txt!!!')
else:
    with open(checkpoint_path+'/result.txt', 'w') as f:
        pass
    for checkpoint in os.listdir(checkpoint_path):
        if 'checkpoint-' in checkpoint:
            print(checkpoint)
            PATH = os.path.join(checkpoint_path, checkpoint)
            # valid
            print('valid')
            if 'pytorch_model.bin-predictions-valid' not in os.listdir(checkpoint_path + '/' + checkpoint):
                b_valid = my_evaluate_parallel(checkpoint_path=PATH, test_or_valid='valid')
            # else:
            #     b = bleu(PATH + "/pytorch_model.bin", 'dev')
            #     print(b)

            # test
            print('test')
            if 'pytorch_model.bin-predictions-test' not in os.listdir(checkpoint_path + '/' + checkpoint):
                b_test = my_evaluate_parallel(checkpoint_path=PATH, test_or_valid='test')
            # else:
            #     b1 = bleu(PATH + "/pytorch_model.bin", 'test')
            #     print(b1)
            with open(checkpoint_path+'/result.txt', 'a+') as f:
                f.write(checkpoint+'\t'+'dev: '+str(b_valid)+ '\t test:'+str(b_test)+ '\n')