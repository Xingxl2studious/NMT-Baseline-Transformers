CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --source_lang th \
    --target_lang en \
    --lang_pair en-th \
    --saved_dir 'checkpoint' \
    --pretrained_model facebook/mbart-large-50\
    --batch_size 16 \
    --max_sentence_length 128\
    --device_num 0