import math
from typing import List
from anytree import Node, RenderTree
import dt_global as G
from math import *
from collections import defaultdict
from itertools import chain
from dt_core import *
from dt_provided import *
import csv
import numpy as np  # numpy==1.19.2
import dt_global


def get_depth(cur):
    if cur.is_leaf:
        return cur.depth
    else:
        return max(get_depth(cur.children[0]), get_depth(cur.children[1]))



def main():
    data = read_data("./data.csv")
    full = Node("root", depth=0)

    split_node(full, data, G.feature_names[:-1])
    print(RenderTree(full))



if __name__ == '__main__':
    main()