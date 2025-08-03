from huggingface_hub import upload_folder

upload_folder(
    repo_id="blakeziegler/qwen3-4b-dapt-v1",
    folder_path="./qwen3_4b_dapt_v1",
    repo_type="model"
)