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

        # All possible actions
        self.actions = game.actions
        self.states = game.states
        self.state_dict = self.generate_state_dict()
        self.hold_all = self.actions[len(self.actions)-1]
        # TODO: Find the optimal gamma and theta values
        gamma = 0.9
        theta = 0.001
        values = {}
        # Populate the values dictionary with the final value for each state
        for state in self.states:
            values[state] = game.final_score(state)
        start = time.process_time()
        self.policy = self.value_iteration(self.states, self.actions, gamma, theta, values)
        end = time.process_time()
        print(self.policy)
        print("initialisation time = ", end-start)

    def generate_state_dict(self):
        # Efficiency!
        state_dict = {}
        for state in self.states:
            for action in self.actions:
                state_dict[(state, action)] = self.game.get_next_states(action, state)
        return state_dict

    def value_iteration(self, states, actions, gamma, theta, values):
        policy = {}
        while True:
            delta = 0
            for state in states:
                temp = values[state] # Represents  V(S) at k-1
                max_value = 0
                best_action = None
                for action in actions:
                    possible_states, game_over, reward, probabilities = self.state_dict[(state, action)]
                    # Only one possible expected value if the action ends the game
                    if game_over:
                        expected_value = reward + (gamma * values[state])
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action
                    # Otherwise, calculate the expected value of taking this action
                    else:
                        # TODO: this may be unnecessary
                        sp = zip(possible_states, probabilities)
                        expected_value = 0
                        # TODO: probably dont need state here
                        # Sum the rewards and probabilities of the action according to the Bellman optimality equation
                        for possible_states, prob in sp:
                            #possible_states, game_over, reward, probabilities = self.state_dict[(possible_states, self.hold_all)]
                            expected_value += prob * (reward + (gamma * values[possible_states])) # Sum the expected value of each possible outcome
                        if expected_value >= max_value:
                            max_value = expected_value
                            best_action = action

                values[state] = max_value
                policy[state] = best_action
                delta = max(delta, abs(temp - max_value))
            # TODO: Maybe can check this earlier?
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

    game = DiceGame()


    #agent1 = AlwaysHoldAgent(game)
    # play_game_with_agent(agent1, game, verbose=True)

    print("\n")

    #agent2 = PerfectionistAgent(game)
    # play_game_with_agent(agent2, game, verbose=True)
    #
    # print("\n")
    #
    agent3 = MyAgent(game)
    n = 0
    total_score = 0
    while n < 100:
        score = play_game_with_agent(agent3, game, verbose=False)
        total_score += score
        n += 1
    print("Average Score = ", total_score / 100)





if __name__ == "__main__":
    main()
# YOUR CODE HERE
