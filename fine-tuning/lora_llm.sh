NPROC_PER_NODE=1\
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen2-audio-7b-instruct \
    --dataset '' \
    --train_type custom \
    --optimizer custom \
    --external_plugins 'examples/train/multimodal/custom_tuner/custom_plugin.py' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --save_only_model true