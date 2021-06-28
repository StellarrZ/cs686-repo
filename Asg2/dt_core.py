# version 1.1
import math
from typing import List
from anytree import Node

import dt_global as G

from math import *
from collections import defaultdict


def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """ 
    indFea = G.feature_names.index(feature)
    ret = []

    table = defaultdict(set)
    for row in examples:
        table[row[indFea]].add(row[G.label_index])
    
    regVal, regLabs = None, set()
    for i, key in enumerate(sorted(table.keys())):
        if i != 0 and (len(regLabs) + len(table[key]) > 2 or table[key] != regLabs):
            ret.append((regVal + key) / 2)
        regVal, regLabs = key, table[key]

    return ret



def choose_feature_split(examples: List, features: List[str]) -> (str, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None and -1.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature and the best split value
    :rtype: str, float
    """   
    def __neg_ent(indFea, midWay):
        # num = 0
        # for i in range(len(examples)):
        #     if examples[i][indFea] <= midWay:
        #         num += 1
        # p = num / len(examples)
        # return round(p * log2(p) + (1 - p) * log2((1 - p)), 6)

        countL, countR = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for i in range(len(examples)):
            if examples[i][indFea] <= midWay:
                countL[examples[i][G.label_index]] += 1
            else:
                countR[examples[i][G.label_index]] += 1
        sumL, sumR = sum(countL.values()), sum(countR.values())
        pLeft = sum(countL.values()) / len(examples)
        pListL = [num / sumL for num in countL.values()]
        pListR = [num / sumR for num in countR.values()]
        pListT = [(countL[key] + countR[key]) / (sumL + sumR) for key in set(countL.keys()).union(countR.keys())]
        
        return round(sum(list(map(lambda p: p * log2(p), pListT))) -
                     sum(list(map(lambda p: p * log2(p), pListL))) * pLeft - 
                     sum(list(map(lambda p: p * log2(p), pListR))) * (1 - pLeft), 6)
    

    regFea, regNegEnt, regMidWay = None, 0, -1
    for fea in features:
        indFea = G.feature_names.index(fea)
        tem = [(__neg_ent(indFea, midWay), midWay) for midWay in get_splits(examples, fea)]
        negEnt, midWay  = min(tem) if tem else (0, -1)
        if negEnt < regNegEnt:
            regFea, regNegEnt, regMidWay = fea, negEnt, midWay

    return regFea, round(regMidWay, 6)



def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """ 
    retLeft, retRight = [], []
    indFea = G.feature_names.index(feature)

    for row in examples:
        if row[indFea] <= split:
            retLeft.append(row.copy())
        else:
            retRight.append(row.copy())

    return retLeft, retRight



def get_majority(examples):
    # count = [0] * G.num_label_values
    # for row in examples:
    #     count[row[G.label_index]] += 1

    # return count.index(max(count))

    count = defaultdict(lambda: 0)
    for row in examples:
        count[row[G.label_index]] -= 1
    
    return min([(count[x], x) for x in count])[1]



def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """ 
    # def __get_majority():
    #     count = [0] * G.num_label_values
    #     for row in examples:
    #         count[row[G.label_index]] += 1

    #     return count.index(max(count))


    major = get_majority(examples)
    if max_depth == 0:
        cur_node.decision = major
        return
    
    splFea, splVal = choose_feature_split(examples, features)
    if not splFea:
        cur_node.decision = major
        return

    cur_node.major, cur_node.numExs = major, len(examples)
    cur_node.feature, cur_node.split = splFea, splVal
    leftExs, rightExs = split_examples(examples, splFea, splVal)
    lchild = Node(" %s<%.3f "%(splFea, splVal), cur_node, depth=cur_node.depth + 1)
    rchild = Node(" %s>%.3f "%(splFea, splVal), cur_node, depth=cur_node.depth + 1)
    split_node(lchild, leftExs, features, max_depth - 1)
    split_node(rchild, rightExs, features, max_depth - 1)
    


def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    root = Node("root", depth=0)
    split_node(root, examples, features, max_depth)
    return root



def predict(cur_node: Node, example, max_depth=math.inf, \
    min_num_examples=0) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.

    If min_num_examples is provided and the number of examples at the node is less than min_num_examples, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the decision for the given example
    :rtype: int
    """ 
    if cur_node.is_leaf:
        return cur_node.decision
    elif max_depth <= 0 or cur_node.numExs < min_num_examples:
        return cur_node.major
    
    indFea = G.feature_names.index(cur_node.feature)
    nextNode = cur_node.children[0] if example[indFea] <= cur_node.split else cur_node.children[1]
    return predict(nextNode, example, max_depth - 1, min_num_examples)



def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf, \
    min_num_examples=0) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth or the min_num_examples, 
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth or min_num_examples.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    accNum = 0
    for row in examples:
        if predict(cur_node, row[:-1], max_depth, min_num_examples) == row[-1]:
            accNum += 1

    return accNum / len(examples)



def post_prune(cur_node: Node, min_num_examples: float):
    """
    Given a tree with cur_node as the root, and the minimum number of examples,
    post prunes the tree using the minimum number of examples criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the number of examples at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the number of examples at every leaf parent is greater than
    or equal to the pre-defined value of the minimum number of examples.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_num_examples: the minimum number of examples
    :type min_num_examples: float
    """
    if cur_node.is_leaf():
        return
    
    post_prune(cur_node.children[0], min_num_examples)
    post_prune(cur_node.children[1], min_num_examples)

    if cur_node.children[0].is_leaf() and \
       cur_node.children[1].is_leaf() and \
       cur_node.numExs < min_num_examples:
       cur_node.decision = cur_node.major
       del cur_node.children
