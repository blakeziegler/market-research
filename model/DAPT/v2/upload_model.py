from huggingface_hub import upload_folder

upload_folder(
    repo_id="blakeziegler/llama_8b_dapt-600k_v1",
    folder_path="./llama_8b_dapt-600k_v1",
    repo_type="model"
)