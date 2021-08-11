from rl_provided import *
import numpy as np
from typing import Tuple, List
from copy import deepcopy as cp


def get_transition_prob(n_sa, n_sas, curr_state: State, dir_intended: int, next_state: State) -> float:
    """
    Determine the transition probability based on counts in n_sa and n_sas'.
    curr_state is s. dir_intended is a. next_state is s'.

    @return: 0 if we haven't visited the state-action pair yet (i.e. N_sa = 0).
      Otherwise, return N_sas' / N_sa.
    """
    return 0 if n_sa[curr_state][dir_intended] == 0 else n_sas[curr_state][dir_intended][next_state] / n_sa[curr_state][dir_intended]


def exp_utils(world, utils, n_sa, n_sas, curr_state: State) -> List[float]:
    """
    @return: The expected utility values for all four possible actions.
    i.e. calculates sum_s'( P(s' | s, a) * U(s')) for all four possible actions.

    The returned list contains the expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the expected utility
    if the agent ends up going down from the current state.
    """
    ret = []
    secondaries = set(get_next_states(world.grid, curr_state))
    for a in range(4):
        utility = 0
        for next_state in secondaries:
            utility += get_transition_prob(n_sa, n_sas, curr_state, a, next_state) * utils[next_state]
        ret.append(utility)
    
    return ret


def optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> List[float]:
    """
    @return: The optimistic expected utility values for all four possible actions.
    i.e. calculates f ( sum_s'( P(s' | s, a) * U(s')), N(s, a) ) for all four possible actions.

    The returned list contains the optimistic expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the optimistic expected utility
    if the agent ends up going down from the current state.
    """
    ret = exp_utils(world, utils, n_sa, n_sas, curr_state)
    for a in range(4):
        if n_sa[curr_state][a] < n_e:
            ret[a] = r_plus
    
    return ret


def update_utils(world, utils, n_sa, n_sas, n_e: int, r_plus: float) -> np.ndarray:
    """
    Update the utility values via value iteration until they converge.
    Call `utils_converged` to check for convergence.
    @return: The updated utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    shape = world.grid.shape
    pre, cur = float("inf"), cp(utils)
    while not utils_converged(pre, cur):
        pre, cur = cur, np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if is_goal(world.grid, (i, j)):
                    cur[i, j] = pre[i, j]
                elif not is_wall(world.grid, (i, j)):
                    exps = optimistic_exp_utils(world, pre, n_sa, n_sas, (i, j), n_e, r_plus)
                    cur[i, j] = world.reward + world.discount * exps[np.argmax(exps)]
    
    return cur


def get_best_action(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> int:
    """
    @return: The best action, based on the agent's current understanding of the world, to perform in `curr_state`.
    """
    return int(np.argmax(optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state, n_e, r_plus)))


def adpa_move(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> Tuple[State, np.ndarray]:
    """
    Execute ADP for one move. This function performs the following steps.
        1. Choose best action based on the utility values (utils).
        2. Make a move by calling `make_move_det`.
        3. Update the counts.
        4. Update the utility values (utils) via value iteration.
        5. Return the new state and the new utilities.

    @return: The state the agent ends up in after performing what it thinks is the best action + the updated
      utilities after performing this action.
    @rtype: A tuple (next_state, next_utils), where:
     - next_utils is an `np.ndarray` of size `world.grid.shape` of type `float`
    """
    a = get_best_action(world, utils, n_sa, n_sas, curr_state, n_e, r_plus)
    next_state = world.make_move_det(a, n_sa)
    n_sa[curr_state][a] += 1
    n_sas[curr_state][a][next_state] += 1
    return next_state, update_utils(world, utils, n_sa, n_sas, n_e, r_plus)


def utils_to_policy(world, utils, n_sa, n_sas) -> np.ndarray:
    """
    @return: The optimal policy derived from the given utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `int`.
    """
    # Initialize the policy.
    policy = np.zeros(world.grid.shape, dtype=int)

    shape = world.grid.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not_goal_nor_wall(world.grid, (i, j)):
                policy[i, j] = int(np.argmax(exp_utils(world, utils, n_sa, n_sas, (i, j))))
    
    return policy


def is_done_exploring(n_sa, grid, n_e: int) -> bool:
    """
    @return: True when the agent has visited each state-action pair at least `n_e` times.
    """
    shape = grid.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not_goal_nor_wall(grid, (i, j)) and np.min(n_sa[i, j]) < n_e:
                return False
    
    return True


def adpa(world_name: str, n_e: int, r_plus: float) -> np.ndarray:
    """
    Perform active ADP. Runs a certain number of moves and returns the learned utilities and policy.
    Stops when the agent is done exploring the world and the utility values have converged.
    Call `utils_converged` to check for convergence.

    Note: By convention, our tests expect the utility of a "wall" state to be 0.

    @param world_name: The name of the world we wish to explore.
    @param n_e: The minimum number of times we wish to see each state-action pair.
    @param r_plus: The maximum reward we can expect to receive in any state.
    @return: The learned utilities.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    # Initialize the world
    world = read_world(world_name)
    grid = world.grid
    num_actions = world.num_actions

    # Initialize persistent variable
    utils = np.zeros(grid.shape)
    n_sa = np.zeros((*grid.shape, num_actions))
    n_sas = np.zeros((*grid.shape, num_actions, *grid.shape))

    shape = grid.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if is_goal(grid, (i, j)):
                utils[i, j] = grid[i, j]

    cur = utils
    while not is_done_exploring(n_sa, grid, n_e):
        a, cur = adpa_move(world, cur, n_sa, n_sas, world.curr_state, n_e, r_plus)

    return cur
