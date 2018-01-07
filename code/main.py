from asag_utils import *

if __name__ == '__main__':
    # print(cur_time())
    # score_answer(fn_prefix='knnr',
    #              reliable=True,
    #              feature='sent2vec',
    #              model='knnr',
    #              model_params={'n_neighbors':5, 'weights':"distance"},
    #              qwise=True,
    #              training_scale=0)
    #
    # score_answer(fn_prefix='cos_l2',
    #              reliable=True,
    #              feature='sent2vec',
    #              model='cos',
    #              model_params={'n_neighbors':5, 'dist_func':'l2'},
    #              qwise=True,
    #              training_scale=0)

    score_answer(fn_prefix='svr_linear',
                 reliable=True,
                 feature='bow_1gram',
                 model='svr',
                 model_params={'kernel': "linear"},
                 qwise=True,
                 training_scale=0)