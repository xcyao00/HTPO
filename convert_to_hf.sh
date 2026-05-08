python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ./exps/ckpts/HTPO/HTPO-Qwen3-8B-Base-multinode/global_step_100/actor \
    --target_dir ./exps/ckpts/HTPO/HTPO/HTPO-Qwen3-8B-Base-multinode/global_step_100/actor/huggingface