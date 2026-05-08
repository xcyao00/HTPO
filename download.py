from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3-8B-Base", cache_dir="exps/models")
snapshot_download("Qwen/Qwen3-8B", cache_dir="exps/models")