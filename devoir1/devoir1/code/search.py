###
# nom: Simon Kaplo           matricule: 1947701
# nom: Bassem Michel Ghaly   matricule: 1951389
###

# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from ctypes.wintypes import PUSHORT
from pydoc import visiblename
from typing import Counter
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 1 ICI
    '''

    startState = problem.getStartState()
    stateStack = util.Stack()
    stateStack.push((startState,[]))
    visitedStates = []
    while not stateStack.isEmpty():
        state = stateStack.pop()
        if state[0] not in visitedStates:
            if problem.isGoalState(state[0]):
                return state[1]  
            else:
                sucessorsTuple = problem.getSuccessors(state[0])                          
                for sucessor in sucessorsTuple:      
                    if sucessor[0] not in visitedStates:
                        path = list(state[1])
                        path.append(sucessor[1])
                        stateStack.push((sucessor[0],path))
                visitedStates.append(state[0])
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    startState = problem.getStartState()
    stateQueue = util.Queue()
    stateQueue.push((startState,[]))
    visitedStates = []
    while not stateQueue.isEmpty():                
        state = stateQueue.pop()
        if state[0] not in visitedStates:
            if problem.isGoalState(state[0]):
                return state[1]  
            else: 
                sucessorsTuple = problem.getSuccessors(state[0])  
                for sucessor in sucessorsTuple:      
                    if sucessor[0] not in visitedStates:
                        path = list(state[1])
                        path.append(sucessor[1])
                        stateQueue.push((sucessor[0],path))
                visitedStates.append(state[0])
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''

    startState = problem.getStartState()
    statesPriorityQueue = util.PriorityQueue()
    statesPriorityQueue.push((startState,[],0),0)
    visitedStates = []

    while statesPriorityQueue.isEmpty() == False:
        state = statesPriorityQueue.pop() 
        if state[0] not in visitedStates:
            if problem.isGoalState(state[0]):
                return state[1]
            else:            
                sucessorsTuple = problem.getSuccessors(state[0]) 
                for sucessor in sucessorsTuple:
                    if sucessor[0] not in visitedStates:
                        path = list(state[1])
                        path.append(sucessor[1])
                        cost = float(state[2] + sucessor[2])
                        statesPriorityQueue.update((sucessor[0],path,cost),cost)
                visitedStates.append(state[0])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 4 ICI
    '''
    startState = problem.getStartState()
    statesPriorityQueue = util.PriorityQueue()
    initalCost = float(heuristic(startState,problem))
    statesPriorityQueue.push((startState,[],0),initalCost)
    visitedStates = []

    while statesPriorityQueue.isEmpty() == False:
        state = statesPriorityQueue.pop() 
        if state[0] not in visitedStates:           
            if problem.isGoalState(state[0]):
                return state[1]
            else: 
                sucessorsTuple = problem.getSuccessors(state[0]) 
                for sucessor in sucessorsTuple:
                    if sucessor[0] not in visitedStates:
                        path = list(state[1])
                        path.append(sucessor[1])
                        cost = float(state[2] + sucessor[2])
                        priority = float(state[2]+sucessor[2]+heuristic(sucessor[0],problem))
                        statesPriorityQueue.update((sucessor[0],path,cost),priority)                
                visitedStates.append(state[0])

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
