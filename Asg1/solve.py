from board import *
from copy import deepcopy as cp
from collections import deque
from heapq import *


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    origin = State(init_board, hfn, hfn(init_board), 0)

    # # debug
    # print(hash(init_board))
    # print(tuple(map(tuple, init_board.grid)))

    hp, mem = [(origin.f, origin.id, 0, origin)], set()
    while hp:
        # # debug
        # print(len(hp))
        # print(nsmallest(min(3, len(hp)), hp))

        _, curId, _, cur = heappop(hp)
        if curId not in mem:
            mem.add(curId)
            if is_goal(cur):
                return get_path(cur), cur.depth
            else:
                for suc in get_successors(cur):
                    heappush(hp, (suc.f, suc.id, curId, suc))
    
    return [], -1

    # raise NotImplementedError



def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    def __index_gcar(cars):
        return next(i for i, car in enumerate(cars) if car.is_goal == True)
    
    
    goalCoord = init_board.size - 2     # pre-defined

    origin = State(init_board, zero_heuristic, 0, 0)
    if is_goal(origin):
        return [origin], 0
    
    st, mem = [origin], set()
    while st:
        # ### debug
        # print(len(st), st[-1].id)
        # st[-1].board.display()

        cur = st.pop()
        if cur.id not in mem:
            mem.add(cur.id)
            if pre_goal(cur):   # tail pruning
                return (get_path(cur) + 
                        [gen_secondary_state(cur, __index_gcar(cur.board.cars), goalCoord)], 
                        cur.depth + 1)
            else:
                st += sorted(get_successors(cur), key=lambda x: -x.id)
    
    return [], -1
    # raise NotImplementedError



def gen_secondary_state(state, carInd, secCoord):
    """
    Self-defined module
    Pre-request: is_goal() == False
    Returns True if the state is `one step away from` the goal state and False otherwise.

    :param state: The current state.
    :type state: State
    :param carInd: Index of the moving car.
    :type carInd: int
    :param secCoord: The secondary coordination after moving.
    :type secCoord: int
    :return: The secondary state.
    :rtype: State
    """
    secondaryCars = cp(state.board.cars)
    secondaryCars[carInd].set_coord(secCoord)
    secondaryBoard = Board(state.board.name, state.board.size, 
                           secondaryCars)

    suc = State(secondaryBoard, state.hfn, 
                state.hfn(secondaryBoard) + state.depth + 1, 
                state.depth + 1)
    
    suc.parent = state
    return suc


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """
    ret = []
    for j, car in enumerate(state.board.cars):

        route = [False] * car.var_coord + [True] * car.length + \
                [False] * (6 - car.var_coord - car.length)
        
        oFlag = car.orientation == 'h'
        for i in range(6):
            if (oFlag and state.board.grid[car.fix_coord][i] == '.' or 
                not oFlag and state.board.grid[i][car.fix_coord] == '.'):
                route[i] = True

        for i in range(7 - car.length):
            section = route[min(i, car.var_coord):max(i, car.var_coord) + car.length]
            if car.var_coord != i and sum(section) == len(section):
                # f, hfn to be edited
                ret.append(gen_secondary_state(state, j, i))

                # ### debug
                # print("A")
                # print(suc.board.cars[j])
                # print(suc.board.cars[j].orientation, suc.board.cars[j].length)
                # print(suc.board.cars[j].var_coord, suc.board.cars[j].fix_coord)
                # print(i, section, route)
                # state.board.display()
                # print("B")
                # suc.board.display()

    return ret

    # raise NotImplementedError


def pre_goal(state):
    """
    Self-defined module
    Pre-request: is_goal() == False
    Returns True if the state is `one step away from` the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """
    for ch in state.board.grid[2][::-1]:
        if ch == '>':
            return True
        elif ch != '.':
            return False


def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """
    return True if state.board.grid[2][-1] == '>' else False

    # raise NotImplementedError


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    ret, cur = deque(), state
    while cur:
        ret.appendleft(cur)
        cur = cur.parent
    
    return list(ret)

    # raise NotImplementedError


def blocking_heuristic(board):
    """
    Returns the heuristic value for the given board
    based on the Blocking Heuristic function.

    Blocking heuristic returns zero at any goal board,
    and returns one plus the number of cars directly
    blocking the goal car in all other states.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    rightEnd = board.grid[2].index('>') + 1
    return 0 if rightEnd == board.size else 7 - rightEnd - board.grid[2][rightEnd:].count('.')

    # raise NotImplementedError


def advanced_heuristic(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    def __extract(indCol):
        raw = [board.grid[x][indCol] for x in range(6)]
        col = [board.grid[x][indCol] != '.' for x in range(6)]

        if raw[1] == '^' and raw[2] == 'v':
            obsLen = 2
            col[1] = False
        elif raw[2] == '^' and raw[3] == 'v':
            obsLen = 2
            col[3] = False
        elif raw[2] == 'v' and raw[1] != '|':
            obsLen = 1
        else:
            obsLen = 3
            if raw[2] == '|':
                col[3] = False
            elif raw[2] == '^':
                col[3], col[4] = False, False

        return obsLen, col

    
    # class PROJ:
    #     def __init__(self):
    #         self.proj = set((2))
        
    #     def set_proj(self, col, low, high):
    #         for x in range(low, high + 1):
    #             if col[x]:
    #                 self.proj.add(x)
        
    #     def to_set(self):
    #         return self.proj


    rightEnd = board.grid[2].index('>') + 1
    if rightEnd == board.size:
        return 0

    """
    3+: ADD 3~5
    2 : ADD min 0~1 , 3~4; priority 3~4
    1 : ADD min 1, 3; priority 3
    """
    ret = 0
    proj = set({2})

    for j, obstacle in enumerate(board.grid[2][rightEnd:], rightEnd):
        if obstacle != '.':
            ret += 1
            obsLen, col = __extract(j)
            if obsLen == 1:
                if col[1] and col[3]:
                    proj.add(3)
            elif obsLen == 2:
                if col[0] + col[1] < col[3] + col[4]:
                    if col[0]:
                        proj.add(0)
                    if col[1]:
                        proj.add(1)
                else:
                    if col[3]:
                        proj.add(3)
                    if col[4]:
                        proj.add(4)
            else:
                for x in range(3, 6):
                    if col[x]:
                        proj.add(x)
    
    ret += len(proj)
    return ret

            # if obstacle == '|':
            #     ret += 3 - col[4:].count('.')
            # elif obstacle == '^':
            #     if board.grid[3][j] == 'v':
            #         ret += min(3 - col[:2].count('.'), 2 - (col[4][j] == '.'))
            #     ret += min(3 - col[:2].count('.'), 2 - min(1, col[3:].count('.')))
            #     pass
            # else:
            #     pass

    # ret = 1

    # for j, ch in enumerate(board.grid[2][rightEnd:], rightEnd):
    #     if ch != '.':
    #         cnt = 0
    #         for i in range(6):
    #             if board.grid[i][j] == '.':
    #                 cnt += 1

    #         if ch == '|':
    #             ret += 3 - min(2, cnt)
    #         else:
    #             ret += 2 - min(1, cnt)
    
    # return ret



# def coincident_heuristic(board):
#     """
#     An advanced heuristic of your own choosing and invention.

#     :param board: The current board.
#     :type board: Board
#     :return: The heuristic value.
#     :rtype: int
#     """
#     rightEnd = board.grid[2].index('>') + 1
#     if rightEnd == board.size:
#         return 0

#     ret = 1
#     for j, ch in enumerate(board.grid[2][rightEnd:], rightEnd):
#         if ch != '.':
#             cnt = 0
#             for i in range(6):
#                 if board.grid[i][j] == '.':
#                     cnt += 1

#             if ch == '|':
#                 ret += 3 - min(2, cnt)
#             else:
#                 ret += 2 - min(1, cnt)
    
#     return ret
