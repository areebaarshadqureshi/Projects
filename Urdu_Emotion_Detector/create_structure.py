from pathlib import Path

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "app",
    "outputs/figures",
    "asserts"
]

files = [
    "CLAUDE.md",
    "README.md",
    "requirements.txt",
    ".gitignore",

    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",

    "notebooks/NB01_data_cleaning.ipynb",
    "notebooks/NB02_eda.ipynb",
    "notebooks/NB03_finetuning.ipynb",
    "notebooks/NB04_explainability.ipynb",
    "notebooks/NB05_bias_audit.ipynb",

    "src/preprocess.py",
    "src/dataset.py",
    "src/model.py",
    "src/train.py",
    "src/predict.py",

    "app/app.py",
    "app/requirements.txt",

    "outputs/classification_report.txt",
    "outputs/error_analysis.md",
    "outputs/bias_audit.md"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch()

print("Project structure created successfully!")