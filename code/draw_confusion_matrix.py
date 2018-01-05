from config import *
from basic_util import plot_confusion_matrix, read_confusion_data, draw_confusion_matrix
from sklearn.metrics import confusion_matrix
WAYS = 11
labels = [str(i/2) for i in range(WAYS)]

if __name__ == '__main__':
    file_list = os.listdir(RESULTS_PATH + '/results')
    print(file_list)
    for f in file_list:
        if f.startswith('.'):
            continue
        if not os.path.exists(RESULTS_PATH + '/results/' + f + '/result.txt'):
            continue
        print(f)
        pres, exps = read_confusion_data(RESULTS_PATH + '/results/' + f + '/result.txt')
        print('pres:', pres)
        print('exps:', exps)
        print()
        data = confusion_matrix(exps, pres)
        plot_confusion_matrix(cm=data, classes=labels, path_name=RESULTS_PATH+'/results/'+f+'/cm.png', normalize=True)
        # draw_confusion_matrix(data, labels, RESULTS_PATH+'/results/'+f+'/cm1.png')