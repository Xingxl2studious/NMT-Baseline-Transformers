CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --source_lang th \
    --target_lang en \
    --lang_pair en-th \
    --saved_dir 'checkpoint' \
    --pretrained_model 'facebook/mbart-large-50'\
    --lr 1e-4 \
    --batch_size 12 \
    --max_sentence_length 128\
    --freeze_decoder 0

# 
