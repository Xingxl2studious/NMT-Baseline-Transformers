import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from source.data import Seq2SeqDataCollator
from torch.utils.data import DataLoader
import evaluate
import pdb

def evaluate_parallel(checkpoint_path, pretrained_model, tokenized_dataset_dict, max_sentence_length, test_or_valid='dev', device_num=1, batch_size=16):
    bleu = evaluate.load("source/sacrebleu")

    # load model
    device = torch.device('cuda:'+str(device_num))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model_config = AutoConfig.from_pretrained(pretrained_model)

    # model process
    translation_model = AutoModelForSeq2SeqLM.from_config(model_config)
    PATH = checkpoint_path + "/pytorch_model.bin"
    translation_model.load_state_dict(torch.load(PATH, map_location={'cuda:1':'cpu','cuda:0':'cpu','cuda:2':'cpu','cuda:3':'cpu','cuda:4':'cpu','cuda:5':'cpu','cuda:6':'cpu','cuda:7':'cpu','cuda:8':'cpu','cuda:9':'cpu'}), strict=False)
    translation_model.to(device)
    
    # load data
    data_collator = Seq2SeqDataCollator(max_sentence_length, tokenizer.pad_token_id, tokenizer.pad_token_id)
    data_loader = DataLoader(tokenized_dataset_dict[test_or_valid], collate_fn=data_collator, batch_size=batch_size)
    
    # generate
    references = []
    predictions = []
    for example in tqdm(data_loader, total=len(data_loader)):
        input_ids = example['input_ids']
        generated_ids = translation_model.generate(input_ids.to(device))
        generated_ids = generated_ids.detach().cpu().numpy()
        prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = example['labels'].detach().cpu().numpy()
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
        references += targets
        predictions += prediction
        
    # compute bleu
    results = bleu.compute(predictions=predictions, references=references)
    print(results)
    
    # save
    with open(f'{PATH}-references-'+test_or_valid, 'w') as f:
        for lines in references:
            for line in lines:
                f.write(line + '\n')
    with open(f'{PATH}-predictions-'+test_or_valid, 'w') as f:
        for line in predictions:
            f.write(line + '\n')   
        
    return results['score']


