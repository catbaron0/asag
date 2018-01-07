import os, sys
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = SCRIPT_PATH + "/../data/ShortAnswerGrading_v2.0/data"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/train"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-questions"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-answers"
DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-domains"
# RESULTS_PATH = SCRIPT_PATH + "/../results_sag"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_uq"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ua"
RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ud"

def remove_scores():
    cur_path = sys.path[0]
    scores_path = DATA_PATH + "/scores/"
    scores = os.listdir(scores_path)
    for score_path in scores:
        me = scores_path + score_path + "/me"
        other = scores_path + score_path + "/other"
        diff = scores_path+ score_path + '/diff'
        print('me: ', me)
        print('other: ', other)
        with open(me, 'r') as fm, open(other, 'r') as fo, open(diff, 'w') as fd:
            score_me = np.array(list(map(float, fm.readlines())))
            score_other = np.array(list(map(float, fo.readlines())))
            fd.writelines('\n'.join(list(map(str, abs(score_me - score_other)))))
if __name__ == '__main__':
    remove_scores()
