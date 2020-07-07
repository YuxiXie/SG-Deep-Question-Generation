import os
import sys

from pprint import pprint

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def text_load(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip().strip('===========================').strip()
    data = data.split('\n===========================\n')
    data = [sample.strip().split('\n') for sample in data]
    gold = [sample[1].strip().split('\t')[1].lower() for sample in data]
    pred = [sample[2].strip().split('\t')[1].lower() for sample in data]

    return gold, pred


if __name__ == "__main__":
    ground_turth, predictions = text_load(sys.argv[1])

    scorers = {
        "Bleu": Bleu(4),
        "Meteor": Meteor(),
        "Rouge": Rouge()
    }

    gts = {}
    res = {}
    if len(predictions) == len(ground_turth):
        for ind, value in enumerate(predictions):
            # print(value)
            res[ind] = [value]

        for ind, value in enumerate(ground_turth):
            gts[ind] = [value]
    else:
        Min_Len = min(len(predictions), len(ground_turth))
        for ind in range(Min_Len):
            res[ind] = [predictions[ind]]
            gts[ind] = [ground_turth[ind]]

    # param gts: Dictionary of reference sentences (id, sentence)
    # param res: Dictionary of hypothesis sentences (id, sentence)

    print('samples: {} / {}'.format(len(res.keys()), len(gts.keys())))

    scores = {}
    for name, scorer in scorers.items():
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    
    pprint(scores)
