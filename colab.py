# Check if HG_ACCESS_TOKEN is defined
if "HG_ACCESS_TOKEN" not in globals():
  HG_ACCESS_TOKEN = input("HG_ACCESS_TOKEN:")

# Installing the required modules
import os

os.system("rm -r sample_data")
os.system("pip install --quiet openai accelerate transformers tokenizers datasets torch sentencepiece pandas numpy tqdm huggingface-cli tiktoken")
os.system("pip freeze > requirements.txt")

# Connecting Huggingface
from huggingface_hub import notebook_login

os.system("git config --global credential.helper store")
if len(HG_ACCESS_TOKEN.strip()) == 0:
  notebook_login(write_permission=True)
else:
  os.system(f"huggingface-cli login --token={HG_ACCESS_TOKEN} --add-to-git-credential")

# Connecting Google Drive
# WARNING: If you want to use a folder from the "Shared with me" folders, then
# create a shortcut for that folder into "My Drive"
from google.colab import drive
drive.mount("/content/drive")
