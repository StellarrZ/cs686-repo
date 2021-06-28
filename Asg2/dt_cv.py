# version 1.1
from typing import List
from itertools import chain

import dt_global as G
from dt_core import *


def gen_full_trees(folds):
    trainList, fullTrees = [], []
    for i in range(len(folds)):
        train = list(chain.from_iterable(folds[:i] + folds[i + 1:]))
        trainList.append(train)
        fullTrees.append(learn_dt(train, G.feature_names[:-1]))
    
    return trainList, fullTrees



def cv(fullTrees, trainList, valiList, para, postSwitch=False):
    maxDepth, minExs = (float("inf"), para) if postSwitch else (para, 0)
    accTrain, accVali = [], []
    for k in range(len(valiList)):
        accTrain.append(get_prediction_accuracy(fullTrees[k], trainList[k], maxDepth, minExs))
        accVali.append(get_prediction_accuracy(fullTrees[k], valiList[k], maxDepth, minExs))
    
    return sum(accTrain) / len(valiList), sum(accVali) / len(valiList)



def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """  
    trainList, fullTrees = gen_full_trees(folds)

    retTrain, retVali = [], []
    for para in value_list:
        temTrain, temVali = cv(fullTrees, trainList, folds, para, postSwitch=False)
        retTrain.append(temTrain)
        retVali.append(temVali)

    return retTrain, retVali



def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 
    trainList, fullTrees = gen_full_trees(folds)

    retTrain, retVali = [], []
    for para in value_list:
        temTrain, temVali = cv(fullTrees, trainList, folds, para, postSwitch=True)
        retTrain.append(temTrain)
        retVali.append(temVali)

    return retTrain, retVali
