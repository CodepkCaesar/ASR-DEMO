from huggingface_hub import snapshot_download

# 目标存储路径
local_model_dir = "./models/paraformer-zh-streaming"

# 从 Hugging Face 下载模型
snapshot_download(
    repo_id="funasr/paraformer-zh-streaming",
    local_dir=local_model_dir,
    local_dir_use_symlinks=False,  # 避免软链接，直接存储文件
    resume_download=True,          # 支持断点续传
)