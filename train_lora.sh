DATASET="/mnt/nfs-share/FlagEvalChat/data/process/ceval_json/test"
# DATASET="LinkSoul/instruction_merge_set"

DATA_CACHE_PATH=""
MODEL_PATH="/mnt/nfs/jyg/Chinese-llama-2-7b-data/Chinese-Llama-2-7b-1.1"

# output_dir="./LinkSoul_checkpoints_llama2_base"

# torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
#     --master_port=25003 \
#         train.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path ${DATASET} \
#         --bf16 True \
#         --output_dir ${output_dir} \
#         --num_train_epochs 1 \
#         --per_device_train_batch_size 4 \
#         --per_device_eval_batch_size 4 \
#         --gradient_accumulation_steps 1 \
#         --evaluation_strategy 'no' \
#         --save_strategy 'steps' \
#         --save_steps 1200 \
#         --save_total_limit 5 \
#         --learning_rate 2e-5 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type cosine \
#         --logging_steps 1 \
#         --fsdp 'full_shard auto_wrap' \
#         --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#         --tf32 True \
#         --model_max_length 4096 \
#         --gradient_checkpointing True \
#         --base True


output_dir="./LinkSoul_checkpoints_llama2_chat_lora_4"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 \
    --master_port=25003 \
        train_lora.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATASET} \
        --bf16 True \
        --output_dir ${output_dir} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy 'no' \
        --save_strategy 'steps' \
        --save_steps 1200 \
        --save_total_limit 5 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False


