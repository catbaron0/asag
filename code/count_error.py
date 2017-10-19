import numpy as np
def count_error(fn = 'svr_all'):
    with open(fn, 'r') as fe, open('error.count', 'w') as ec,\
            open('error_abs.count', 'w') as eac,\
            open('error_round.count', 'w') as erc:
        svr_all = map(lambda line:line.split(':'), fn.readlines())
        _, _, _, error, error_abs, error_round = zip(*svr_all)
        count = len(error)
        error = map(float, error)
        error_abs = map(float, error_abs)
        error_round = map(float, error_round)
        def count_hist(f, error):
            # -5~-4, -4~-3
            k = list(np.arange(-4.5, 5.1, 0.5))
            v = [0] * 20
            hist = dict(zip(k,v))
            for e in error:
                for k in hist:
                    if e <= k:
                        hist[k] += 1
                        break
            for k,v in hist.items():
                f.write("{}:{}\n".format(k, v))
            f.write('{}:{}'.format(count, sum(dict.values())))

        count_hist(ec, error)
        count_hist(eac, error_abs)
        count_hist(erc, error_round)
