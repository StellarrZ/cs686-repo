import sys
import unittest

from board import *
from solve import *


def main():
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case01.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case02.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case03.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case43.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case05.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case06.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case07.txt")
    # boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/tst_case10.txt")
    boards = from_file("/home/z7sheng/CS686/cs686-repo/Asg1/jams_posted.txt")
    for k, b in enumerate(boards, 1):
        # print(b.name)
        # print(b.grid)
        # b.display()


        # pathDfs, costDfs = dfs(b)
        # print(b.name, costDfs)

        # print(len(pathDfs))
        # for i, state in enumerate(pathDfs, 1):
        #     print(k, i, "/", len(pathDfs))
        #     state.board.display()


        # pathHeu, costHeu = a_star(b, blocking_heuristic)
        # print(b.name, costHeu)

        # print(len(pathHeu))
        # for i, state in enumerate(pathHeu, 1):
        #     print(k, i, "/", len(pathHeu), "  ", state.f - state.depth, blocking_heuristic(state.board))
        #     state.board.display()


        pathHeu, costHeu = a_star(b, advanced_heuristic)
        print(b.name, costHeu)

        print(len(pathHeu))
        for i, state in enumerate(pathHeu, 1):
            print(k, i, "/", len(pathHeu), "  ", state.f - state.depth, advanced_heuristic(state.board))
            state.board.display()



class MyTest(unittest.TestCase):
    def test_upper(self):
        """Test the upper() function of class string"""
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        """Test isupper() function of class string"""
        self.assertTrue('FOO'.isupper())
        self.assertFalse('foo'.isupper())
        self.assertFalse('foo'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_failing(self):
        """A test that fails"""
        self.assertEqual(True, False)



if __name__ == '__main__':
    main()
    # unittest.main()