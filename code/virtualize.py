from config import *
from basic_util import text_weight_color, clean_text

'''
read questions and answers, generate a html to virtualize them.
'''

# read word-weight dict
# read questions
with open(RAW_PATH + '/questions') as f_questions:
    for q in f_questions:
        q_id = q.split(' ')[0]

        # read word-weight dict
        weight_dicts = []
        with open(RESULTS_PATH + "/word_weights/" + q_id) as f_weights:
            for line in f_weights:
                weights = dict([(item.split(':')[0], float(item.split(':')[1])) for item in line.split(',')])
                weight_dicts.append(weights)

        # read scores
        with open(DATA_PATH + '/scores/{}/ave'.format(q_id), 'r') as f_score:
            scores = f_score.readlines()

        # read answers
        for a_idx in range(len(weight_dicts)):
        # for weight_dict in weight_dicts:
            weight_dict = weight_dicts[a_idx]
            html = ['<html>\n', '<div>'+q+'</div><br>',]
            with open(RAW_PATH_STU + '/' + q_id, encoding='utf-8', errors="ignore") as f_answers:
                answers = f_answers.readlines()
                for i in range(len(answers)):
                    score = scores[i].strip()
                    ans = answers[i].replace('<br>', '')
                    ans = "{}.{}({}) {}".format(q_id, i+1, score, ans.strip()[ans.find(' ')+1:])
                    html.append(text_weight_color(ans, weight_dict)+'\n')
            html.append('</html>')
            path_html = RESULTS_PATH + "/html"
            if not os.path.exists(path_html):
                os.mkdir(path_html)
            with open("{}/{}.{}.html".format(path_html, q_id, a_idx), 'w', encoding='utf-8', errors="ignore") as f_html:
            # with open(path_html + '/' + q_id + '.html', 'w', encoding='utf-8', errors="ignore") as f_html:
                f_html.writelines(html)

