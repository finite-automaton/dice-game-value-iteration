from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np
import time


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class MyAgent(DiceGameAgent):
    def __init__(self, game):
        """
        if your code does any pre-processing on the game, you can do it here

        e.g. you could do the value iteration algorithm here once, store the policy,
        and then use it in the play method

        you can always access the game with self.game
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game)
        start = time.process_time()

        self.actions = game.actions  # All possible actions for the game
        self.states = game.states  # All possible states for the game
        self.state_map = self.generate_state_map()  # Map (dict) of all states, similar to a transition function
        gamma = 1  # Gamma may be 1 because there is no reward for continuing to play, it will always converge
        theta = 0.001  # Trial and error shows this to be an efficient value for generating the optimal policy
        self.policy = self.get_optimal_policy_with_value_iteration(gamma, theta)
        end = time.process_time()
        print(self.policy)
        print("initialisation time = ", end - start)

    def generate_state_map(self):
        """
        Generating this once on initialisation is much more efficient that calling the method on each iteration

        :return: a dictionary of mapping all state and action pairs to their possible outcomes.
        """

        state_dict = {}
        for state in self.states:
            for action in self.actions:
                possible_states, game_over, reward, probabilities = self.game.get_next_states(action, state)
                state_probabilities = zip(possible_states, probabilities)
                state_dict[(action, state)] = (game_over, reward, state_probabilities)
        return state_dict

    def get_optimal_policy_with_value_iteration(self, gamma, theta):
        """

        :param gamma: Gamma value for bellman equation
        :param theta: Theta threshold to stop value iteration
        :return: a dictionary representing a deterministic optimal policy
        """
        values = {}  # Map of state to its expected value
        policy = {}  # Deterministic policy mapping state to action

        # Initialise the values array arbitrarily for all states
        for state in self.states:
            values[state] = 0

        # Iterate over the values until convergence
        while True:
            delta = 0
            for state in self.states:
                temp = values[state]
                max_value = 0  # Holds the max possible expected value of the state
                best_action = None  # Hold the action that is most likely to yield the max exp. value
                for action in self.actions:
                    # Look up the possible states that result from this action
                    game_over, reward, state_probabilities = self.state_map[(action, state)]
                    # If the action ends the game, there is a deterministic reward and no future rewards
                    if game_over:
                        expected_value = reward
                        # If this yields the best expected value, it is the max value and best action
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action
                    # Otherwise, calculate the expected value of taking this action with value iteration
                    else:
                        expected_value = 0  # Initialise expected value so we can sum onto it
                        # Sum the rewards and probabilities of the action according to the Bellman optimality equation
                        for possible_state, prob in state_probabilities:
                            expected_value += prob * (reward + (gamma * values[possible_state]))
                        # If this yields the best expected value, it is the max value and best action
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action
                # Update the policy and state -> expected value map with the current optimal choices
                values[state] = max_value
                policy[state] = best_action
                # Update delta if the absolute difference between the new and old value is larger than current delta
                delta = max(delta, abs(temp - max_value))
            # After each sweep, if delta is less than theta, stop the loop and return the policy
            if delta < theta:
                return policy

    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice

        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions

        read the code in dicegame.py to learn more

        """

        return self.policy[state]


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    if (verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if (verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if (verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if (verbose and not game_over): print(f"Dice: \t\t{state}")

    if (verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run
    np.random.seed(1)

    # seed = 0
    # results = []
    # while seed < 20:
    #     result = runAGameXTimes(seed, 100000)
    #     results.append((seed, result))
    #     seed += 1
    #
    # for result in results:
    #     print("Seed : ", seed, ". Result: ", result)
    game = DiceGame()

    agent1 = AlwaysHoldAgent(game)
    # play_game_with_agent(agent1, game, verbose=True)

    print("\n")

    agent2 = PerfectionistAgent(game)
    # play_game_with_agent(agent2, game, verbose=True)
    #
    # print("\n")
    #
    agent3 = MyAgent(game)
    n = 0
    j = 10000
    total_score = 0
    low = 100
    high = 0
    while n < j:
        score = play_game_with_agent(agent3, game, verbose=False)
        total_score += score
        if score > high:
            high = score
        if score < low:
            low = score
        n += 1


def runAGameXTimes(seed, iterations):
    np.random.seed(seed)
    game = DiceGame()
    agent = MyAgent(game)
    n = 0
    low = 100
    high = 0
    total_score = 0
    while n < iterations:
        score = play_game_with_agent(agent, game, verbose=False)
        total_score += score
        if score > high:
            high = score
        if score < low:
            low = score
        n += 1
    return total_score / iterations

    # print("total Score = ", total_score)
    # print("Average Score = ", total_score / j)
    # print("Highest = ", high)
    # print("Lowest = ", low)


if __name__ == "__main__":
    main()
# YOUR CODE HERE
