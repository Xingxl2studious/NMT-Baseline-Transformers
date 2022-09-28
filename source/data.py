import numpy as np
import torch

# data structure
class Seq2SeqDataCollator:
    
    def __init__(
        self,
        max_length: int,
        pad_token_id: int,
        pad_label_token_id: int = -100
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_label_token_id = pad_label_token_id
        
    def __call__(self, batch):
        source_batch = []
        source_len = []
        target_batch = []
        target_len = []
        for example in batch:
            source = example['input_ids']
            source_len.append(len(source))
            source_batch.append(source)

            target = example['labels']
            target_len.append(len(target))
            target_batch.append(target)

        source_padded = self.process_encoded_text(source_batch, source_len, self.pad_token_id)
        target_padded = self.process_encoded_text(target_batch, target_len, self.pad_label_token_id)
        attention_mask = generate_attention_mask(source_padded, self.pad_token_id)
        return {
            'input_ids': source_padded,
            'labels': target_padded,
            'attention_mask': attention_mask
        }

    def process_encoded_text(self, sequences, sequences_len, pad_token_id):
        sequences_max_len = np.max(sequences_len)
        max_length = min(sequences_max_len, self.max_length)
        padded_sequences = pad_sequences(sequences, max_length, pad_token_id)
        return torch.LongTensor(padded_sequences)


def generate_attention_mask(input_ids, pad_token_id):
    return (input_ids != pad_token_id).long()

    
def pad_sequences(sequences, max_length, pad_token_id):
    """
    Pad the list of sequences (numerical token ids) to the same length.
    Sequence that are shorter than the specified ``max_len`` will be appended
    with the specified ``pad_token_id``. Those that are longer will be truncated.

    Parameters
    ----------
    sequences : list[int]
        List of numerical token ids.

    max_length : int
         Maximum length that all sequences will be truncated/padded to.

    pad_token_id : int
        Padding token index.

    Returns
    -------
    padded_sequences : 1d ndarray
    """
    num_samples = len(sequences)
    padded_sequences = np.full((num_samples, max_length), pad_token_id)
    for i, sequence in enumerate(sequences):
        sequence = np.array(sequence)[:max_length]
        padded_sequences[i, :len(sequence)] = sequence

    return padded_sequences




# method
def batch_tokenize_fn(examples, source_lang, target_lang, pretrained_tokenizer, max_source_length, max_target_length):
    """
    Generate the input_ids and labels field for huggingface dataset/dataset dict.
    
    Truncation is enabled, so we cap the sentence to the max length, padding will be done later
    in a data collator, so pad examples to the longest length in the batch and not the whole dataset.
    """
    sources = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = pretrained_tokenizer(sources, max_length=max_source_length, truncation=True)

    # setup the tokenizer for targets,
    # huggingface expects the target tokenized ids to be stored in the labels field

    # 如果预训练模型源语言目标语言词表不一致，则使用如下方法
    # with pretrained_tokenizer.as_target_tokenizer():
    #     labels = pretrained_tokenizer(targets, max_length=max_target_length, truncation=True)
    
    labels = pretrained_tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def create_translation_data(
    source_input_path: str,
    target_input_path: str,
    output_path: str,
    delimiter: str = '\t',
    encoding: str = 'utf-8'
):
    """
    Creates the paired source and target dataset from the separated ones.
    e.g. creates `train.tsv` from `train.de` and `train.en`
    """
    with open(source_input_path, encoding=encoding) as f_source_in, \
         open(target_input_path, encoding=encoding) as f_target_in, \
         open(output_path, 'w', encoding=encoding) as f_out:

        for source_raw in f_source_in:
            source_raw = source_raw.strip().replace('\t',' ')
            target_raw = f_target_in.readline().strip().replace('\t',' ')
            if source_raw and target_raw:
                output_line = source_raw + delimiter + target_raw + '\n'
                f_out.write(output_line)




