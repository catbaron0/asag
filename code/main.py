from asag_utils import *

if __name__ == '__main__':
    # print(cur_time())
    # for i in range(1,6):
    score_answer(fn_prefix='knnc',
                 reliable=True,
                 feature='sent2vec',
                 model='knnc',
                 model_params={'n_neighbors':5, 'weights':"distance"},
                 qwise=True,
                 training_scale=0)

    score_answer(fn_prefix='cosc',
                 reliable=True,
                 feature='sent2vec',
                 model='cosc',
                 model_params={'n_neighbors':5, 'dist_func':'l2'},
                 qwise=True,
                 training_scale=0)

    score_answer(fn_prefix='svr_linear',
                 reliable=True,
                 feature='sent2vec',
                 model='svr',
                 model_params={'kernel': "linear"},
                 qwise=True,
                 training_scale=0)

    # score_answer(fn_prefix='svr_linear',
    #              reliable=True,
    #              feature='bow_1gram',
    #              model='svr',
    #              model_params={'kernel': "linear"},
    #              qwise=True,
    #              training_scale=0)
