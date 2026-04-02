import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(PROCESSED_DATA_DIR, "submission.csv")

# Pipeline Columns
OHE_COLS = [
    'Cleaned_Degree', 'Family History of Mental Illness', 
    'Cleaned_Profession', 'Have you ever had suicidal thoughts ?', 
    'Working Professional or Student', 'Gender'
]
ORDINAL_COLS = ['Dietary Habits', 'Cleaned_Sleep_Duration']