import os

LEVENSHTEIN = 3

# Paths
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = SCRIPT_PATH + "/../data/XCSD_6Ways/data"
# DATA_PATH = SCRIPT_PATH + "/../data/XCSD_2Ways/data"
# DATA_PATH = SCRIPT_PATH + "/../data/beetle_2Ways/all"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/train"
# DATA_PATH = SCRIPT_PATH + "/../data/kaggle_train"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-questions"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-answers"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-domains"
RESULTS_PATH = SCRIPT_PATH + "/../results_sag"
# RESULTS_PATH = SCRIPT_PATH + "/../results_xcsd_2ways"
# RESULTS_PATH = SCRIPT_PATH + "/../results_beetle_2Ways"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_kaggle_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_uq"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ua"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ud"

RAW_PATH = DATA_PATH + "/raw"
RAW_PATH_STU = DATA_PATH + "/raw/ans_stu"
W2V_PATH = SCRIPT_PATH + '/../data/glove.6B'
W2V_FILE = 'glove.6B.300d.txt'

WEIGHTS_PATH = RESULTS_PATH + "/word_weights"