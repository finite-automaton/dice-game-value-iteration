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
        # Optimal gamma and theta values for the 3 fair six sided dice game
        # gamma = 0.9412
        # theta = 1.251
        gamma = 0.9
        # theta = 1.57
        theta = 0.001

        # self.policy = self.get_optimal_policy_with_value_iteration(gamma, theta)
        self.policy = self.get_optimal_policy_with_policy_iteration(gamma, theta)
        end = time.process_time()
        print(self.policy)
        print("initialisation time = ", end - start)

    def generate_state_map(self):
        # Efficiency!
        state_dict = {}
        for state in self.states:
            for action in self.actions:
                state_dict[(state, action)] = self.game.get_next_states(action, state)
        return state_dict

    def get_optimal_policy_with_policy_iteration(self, gamma, theta):
        # Initialisation
        policy = {}
        values = {}
        for state in self.states:
            policy[state] = ()
            values[state] = 0
        # Policy evaluation
        self.policy_evaluation(gamma, policy, theta, values)
        # Policy Improvement
        policy_stable = True
        while policy_stable:

            for state in self.states:
                old_action = policy[state]
                max_value = 0
                # pi(s) <- argmax...
                best_action = None
                for action in self.actions:
                    possible_states, game_over, reward, probabilities = self.state_map[(state, action)]
                    # Only one possible expected value if the action ends the game
                    if game_over:
                        expected_value = reward
                        # expected_value = reward + (gamma * self.values[state])
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action
                    # Otherwise, calculate the expected value of taking this action
                    else:
                        state_probabilities = zip(possible_states, probabilities)
                        expected_value = 0
                        # Sum the rewards and probabilities of the action according to the Bellman optimality equation
                        for possible_state, prob in state_probabilities:
                            expected_value += prob * (reward + (gamma * values[
                                possible_state]))  # Sum the expected value of each possible outcome
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action
                policy[state] = best_action
                # check
                if old_action != policy[state]:
                    policy_stable = False
                else:
                    self.policy_evaluation(gamma, policy, theta, values)
        return policy

    def policy_evaluation(self, gamma, policy, theta, values):
        while True:
            delta = 0
            for state in self.states:
                old_value = values[state]
                action = policy[state]
                possible_states, game_over, reward, probabilities = self.state_map[(state, action)]
                new_value = 0
                # Only one possible expected value if the action ends the game
                if game_over:
                    new_value = reward
                else:
                    state_probabilities = zip(possible_states, probabilities)
                    expected_value = 0
                    # Sum the rewards and probabilities of the action according to the Bellman optimality equation
                    for possible_state, prob in state_probabilities:
                        expected_value += prob * (reward + (gamma * values[possible_state]))  # Sum the expected value of each possible outcome
                    new_value = expected_value
                values[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:
                break

    def get_optimal_policy_with_value_iteration(self, gamma, theta):
        # Populate the values dict with the arbitrary starting value
        values = {}
        for state in self.states:
            values[state] = 0
        policy = {}
        while True:
            delta = 0
            for state in self.states:
                temp = values[state]  # Represents  V(S) at k-1
                max_value = 0
                best_action = None
                for action in self.actions:
                    possible_states, game_over, reward, probabilities = self.state_map[(state, action)]
                    # Only one possible expected value if the action ends the game
                    if game_over:
                        expected_value = reward
                        # expected_value = reward + (gamma * self.values[state])
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action
                    # Otherwise, calculate the expected value of taking this action
                    else:
                        state_probabilities = zip(possible_states, probabilities)
                        expected_value = 0
                        # Sum the rewards and probabilities of the action according to the Bellman optimality equation
                        for possible_state, prob in state_probabilities:
                            expected_value += prob * (reward + (gamma * values[
                                possible_state]))  # Sum the expected value of each possible outcome
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action

                values[state] = max_value
                policy[state] = best_action
                delta = max(delta, abs(temp - max_value))

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

    game = DiceGame(dice=4, sides=6, penalty=1)

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

    print("total Score = ", total_score)
    print("Average Score = ", total_score / j)
    print("Highest = ", high)
    print("Lowest = ", low)


if __name__ == "__main__":
    main()
# YOUR CODE HERE
