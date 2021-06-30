import argparse
import time
from math import *
from anytree import Node, RenderTree
from typing import List
import dt_global as G
from dt_provided import *
from dt_core import *
from dt_cv import *



def get_depth(cur):
    if cur.is_leaf:
        return cur.depth
    else:
        return max(get_depth(cur.children[0]), get_depth(cur.children[1]))



def main():
    data = read_data("./data.csv")
    folds = preprocess(data)

    if not args.prof:
        full = learn_dt(data, G.feature_names[:-1])
        print(RenderTree(full))
    else:
        print("\nProfiling Results")
        print("========================================")
        start = time.time()
        fullProf = learn_dt(data, G.feature_names[:-1])
        end = time.time()
        print("  Generate fullTree:        %.6f s"%(end - start))

        start = time.time()
        trainAcc_pre, valiAcc_pre = cv_pre_prune(folds, list(range(31)))
        end = time.time()
        print("  CV pre-prune [0:1:31]:    %.6f s"%(end - start))

        start = time.time()
        trainAcc_post, valiAcc_post = cv_post_prune(folds, list(range(0, 301, 20)))
        end = time.time()
        print("  CV post-prune [0:20:301]: %.6f s"%(end - start))
        print("========================================")

        print("\nTuning Results")
        print("========================================")
        print("  Best maxDepth:   ", valiAcc_pre.index(max(valiAcc_pre)))
        print("  Vali-Accuracy:   ", max(valiAcc_pre))
        print("  Best minExamples:", valiAcc_post.index(max(valiAcc_post)) * 20)
        print("  Vali-Accuracy:   ", max(valiAcc_post))
        print("========================================")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prof", type=int, default=0)
    args = parser.parse_args()
    main()