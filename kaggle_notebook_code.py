from kaggle_secrets import UserSecretsClient
from urllib.parse import quote
from pathlib import Path
import subprocess
import shutil
import os

# ---------- settings ----------
GITHUB_USER = "abhayaggOR"
REPO_NAME = "cv_project"
BRANCH = "main"
COMMIT_MSG = "Push Kaggle notebook code and outputs"

REPO_DIR = Path("/kaggle/working/repo_clone")

# Save current notebook input history as a Python file
NOTEBOOK_CODE_FILE = Path("/kaggle/working/kaggle_notebook_code.py")
get_ipython().run_line_magic("history", f"-f {NOTEBOOK_CODE_FILE}")

# Add here whatever you want to push
FILES_TO_PUSH = [
    "/kaggle/working/kaggle_notebook_code.py",
    "/kaggle/working/exported_models/helmet_plate_best.pt",
]

DIRS_TO_PUSH = [
    "/kaggle/working/helmet_plate_yolo11n",   # training outputs
    "/kaggle/working/exported_models",        # exported model folder
]

# ---------- auth ----------
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
auth_url = f"https://{GITHUB_USER}:{quote(token)}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

# ---------- fresh clone ----------
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(["git", "clone", auth_url, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "Kaggle Bot"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.email", "kaggle-bot@example.com"], check=True)

# ---------- copy files ----------
def copy_file(src_path, repo_dir):
    src = Path(src_path)
    if src.exists() and src.is_file():
        dst = repo_dir / src.name
        shutil.copy2(src, dst)
        print("Copied file:", src, "->", dst)
    else:
        print("Skipped missing file:", src)

def copy_dir(src_path, repo_dir):
    src = Path(src_path)
    if src.exists() and src.is_dir():
        dst = repo_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("Copied dir :", src, "->", dst)
    else:
        print("Skipped missing dir :", src)

for f in FILES_TO_PUSH:
    copy_file(f, REPO_DIR)

for d in DIRS_TO_PUSH:
    copy_dir(d, REPO_DIR)

# ---------- git add / commit / push ----------
subprocess.run(["git", "-C", str(REPO_DIR), "add", "."], check=True)

commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "commit", "-m", COMMIT_MSG],
    capture_output=True,
    text=True
)

print(commit.stdout)
print(commit.stderr)

if commit.returncode == 0:
    subprocess.run(["git", "-C", str(REPO_DIR), "push", "origin", BRANCH], check=True)
    print("Push successful.")
else:
    print("No new changes to commit.")
