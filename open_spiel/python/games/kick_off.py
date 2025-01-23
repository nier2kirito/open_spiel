# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kick Off Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np
import random
import pyspiel
from open_spiel.python.utils.poker_evaluator import PokerEvaluator

evaluator = PokerEvaluator()

class Action(enum.IntEnum):
    FOLD = 0
    POST_SB = 1
    POST_BB = 2
    CALL = 3
    BET_1_5 = 4
    BET_3 = 5
    BET_5 = 6
    RAISE_2_5 = 7
    RAISE_5 = 8
    RAISE_8 = 9
    ALL_IN = 10
    CHECK = 11


# Define a dictionary mapping each action to its corresponding amount
ACTION_AMOUNTS = {
    Action.FOLD: 0,
    Action.POST_SB : 0.5,
    Action.POST_BB : 1.0,
    Action.BET_1_5: 1.5,
    Action.BET_3: 3,
    Action.BET_5: 5,
    Action.RAISE_2_5: 2.5,
    Action.RAISE_5: 5,
    Action.RAISE_8: 8,
}


# Add a property to fetch the amount
def amount(action):
    return ACTION_AMOUNTS[action]


_INITIAL_STACK = 20

_NUM_PLAYERS = 4
_DECK = frozenset(
    (rank, suit) for rank in
    ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    for suit in ['h', 'd', 'c', 's'
                 ]  #h : hearts, d: diamonds, c : clubs, s : spades
)
deck_list = list(_DECK)
random.shuffle(deck_list)

_GAME_TYPE = pyspiel.GameType(
    short_name="kick_off",
    long_name="Kick Off",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)

_GAME_INFO = pyspiel.GameInfo(num_distinct_actions=len(Action),
                              max_chance_outcomes=len(_DECK),
                              num_players=_NUM_PLAYERS,
                              min_utility=-_INITIAL_STACK,
                              max_utility=_INITIAL_STACK,
                              utility_sum=0.0,
                              max_game_length=64)


class KickOffGame(pyspiel.Game):
    """A Python version of KickOff poker."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return KickOffState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return KickOffObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
            params)


class KickOffState(pyspiel.State):
    """A python version of the kick off poker state."""

    def __init__(self, game):
        super().__init__(game)
        self.cards = []
        self.bets = [0] * _NUM_PLAYERS
        self.pot = [0.0] * _NUM_PLAYERS
        self.players_stack = [_INITIAL_STACK] * _NUM_PLAYERS
        self._game_over = False
        self._next_player = 0 
        self._round = "preflop"
        self._active_players = set(range(_NUM_PLAYERS))
        self._community_cards = []
        self.cumulative_pot = [0.0] * _NUM_PLAYERS
        # Initialize current bet
        self._current_bet = 0  # Add this line
        self._deck = deck_list.copy()

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every sequential-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self.is_chance_node():
            # Dealing hole cards to players
            return pyspiel.PlayerId.CHANCE
        else:
            return self._next_player

    def _is_betting_round_complete(self):
        """Returns True if all active players have matched the current bet."""
        for player in self._active_players:
            if self.pot[player] < self._current_bet:
                return False  # At least one player hasn't matched the bet
        return True
    def is_chance_node(self):
        """Returns True if the current state is a chance node (e.g., dealing cards)."""
        # A chance node occurs when cards are being dealt
        if self._round == "preflop" and len(self.cards) < _NUM_PLAYERS * 2:
            self._next_player = min(self._active_players)
            return True  # Dealing hole cards to players
        elif self._round in ["flop", "turn", "river"] and len(self._community_cards) < {
            "flop": 3,
            "turn": 4,
            "river": 5,
        }[self._round]:
            self._next_player = min(self._active_players)
            return True  # Dealing community cards
        else:
            return False  # Not a chance node
    def _advance_to_next_player(self):
        """Move to the next player, cycling back to 0 if at the end."""
        if self._round == 'preflop':
            self._next_player = (self._next_player + 1) % _NUM_PLAYERS
        else:
            # Get the sorted list of active players
            active_players_sorted = sorted(self._active_players)
            
            # Find the index of the current next player in the sorted list
            current_index = active_players_sorted.index(self._next_player)
            next_index = (current_index + 1) % len(active_players_sorted)
            self._next_player = active_players_sorted[next_index]
    def information_state_string(self):
        """Returns information state for the CURRENT player."""
        player = self.current_player()
        if not isinstance(player, int) or player < 0 or player >= _NUM_PLAYERS:
            return "No information state for chance/terminal nodes"
        
        # Build information state components
        components = []
        
        # Add private cards for players
        components.append(f"Cards: {self.cards}")
        
        # Add community cards
        components.append(f"Community cards: {self._community_cards}")
        
        # Add betting history
        components.append(f"Pot: {self.pot}")
        components.append(f"Current bet: {self._current_bet}")
        
        return "\n".join(components)
    
    def _deal_cards(self):
        """Deals the hole cards to players and the community cards."""
        if self._round == "preflop" and len(self.cards) < _NUM_PLAYERS * 2:
            # Deal 2 hole cards to each player (total 8 cards for 4 players)
            self.cards = [self._deck.pop() for _ in range(_NUM_PLAYERS * 2)]

        elif self._round == "flop" and len(self._community_cards) == 0:
            # Deal Flop (3 community cards)
            self._community_cards.extend([self._deck.pop() for _ in range(3)])

        elif self._round == "turn" and len(self._community_cards) == 3:
            # Deal Turn (1 community card)
            self._community_cards.append(self._deck.pop())

        elif self._round == "river" and len(self._community_cards) == 4:
            # Deal River (1 community card)
            self._community_cards.append(self._deck.pop())


    def _update_pot(self):
        """Update the pot with the current player's bet."""
        self.pot += self.bets[self._next_player]

    def legal_actions(self):
        """Returns a list of legal actions for the current player."""
        # Handle cases where the state is not a decision node
        if self.is_chance_node() or self.is_terminal():
            return []
        
        player = self.current_player()
        # Ensure player is a valid integer index
        if not isinstance(player, int) or player < 0 or player >= _NUM_PLAYERS:
            return []
        
        legal_actions = [int(Action.FOLD)]

        if self._current_bet == 0 :
            legal_actions.pop()
            legal_actions.append(int(Action.CHECK))
        if self._current_bet > 0 and self.players_stack[player] >= self._current_bet:
            legal_actions.append(int(Action.CALL))
        if self._round == "preflop" and self.current_player() == 0:
            legal_actions.append(int(Action.POST_SB))
        if self._round == "preflop" and self.current_player() == 1:
            legal_actions.append(int(Action.POST_BB))
        if self._round != "preflop":
            if self.players_stack[player] > 1.5:
                legal_actions.append(int(Action.BET_1_5))
            if self.players_stack[player] > 3:
                legal_actions.append(int(Action.BET_3))
            if self.players_stack[player] > 5:
                legal_actions.append(int(Action.BET_5))

        if self._current_bet > 0:
            if self.players_stack[player] > 2.5:
                legal_actions.append(int(Action.RAISE_2_5))
            if self.players_stack[player] > 5:
                legal_actions.append(int(Action.RAISE_5))
            if self.players_stack[player] > 8:
                legal_actions.append(int(Action.RAISE_8))

        if self.players_stack[player] > 0:
            legal_actions.append(int(Action.ALL_IN))

        return legal_actions
    def chance_outcomes(self):
        """
    Returns the possible chance outcomes and their probabilities.
    This determines the next card to be dealt from the remaining deck.
    """
        assert self.is_chance_node(
        ), "This method should only be called at chance nodes."

        # Remaining cards in the deck
        remaining_deck = sorted(_DECK - set(self.cards))

        # Compute the probability for each outcome
        num_outcomes = len(remaining_deck)
        if num_outcomes == 0:
            raise ValueError("No cards left in the deck for chance outcomes.")

        probability = 1.0 / num_outcomes
        return [(card, float(probability)) for card in remaining_deck]

    def _apply_action(self, action):
        if self.is_chance_node():
            self._deal_cards()
        elif action not in self.legal_actions():
            raise ValueError(f"Action {action} is not valid for player {self.current_player()}")
        else:
            # Update the player's bet in the bets list (assuming self.bets tracks current bets)
            # Note: The original code's handling of self.bets might need further adjustment
            # This is a placeholder to fix variable references
            self.bets[self.current_player()] = action  # Changed from append to index assignment
            if action == Action.FOLD:
                # Remove player from active players
                self._active_players.discard(self.current_player())
                # Check if only one player remains
                if len(self._active_players) == 1:
                    self._game_over = True
            elif action == Action.POST_SB:
                self._current_bet = 0.5
                self.pot[self.current_player()] = self._current_bet
                self.players_stack[self.current_player()] -= self._current_bet
            elif action == Action.POST_BB:
                self._current_bet = 1
                self.pot[self.current_player()] = self._current_bet
                self.players_stack[self.current_player()] -= self._current_bet
            elif action == Action.CALL:
                bet_amount = self._current_bet - self.pot[self.current_player()]
                self.pot[self.current_player()] = self._current_bet
                self.players_stack[self.current_player()] -= bet_amount
            elif action in {Action.BET_1_5, Action.BET_3, Action.BET_5}:
                bet_amount = ACTION_AMOUNTS[action]
                self._current_bet = bet_amount
                bet_amount = self._current_bet - self.pot[self.current_player()]
                self.pot[self.current_player()] = self._current_bet
                self.players_stack[self.current_player()] -= bet_amount
            elif action in {Action.RAISE_2_5, Action.RAISE_5, Action.RAISE_8}:
                bet_amount = ACTION_AMOUNTS[action]
                self._current_bet = bet_amount
                bet_amount = self._current_bet - self.pot[self.current_player()]
                self.pot[self.current_player()] = self._current_bet
                self.players_stack[self.current_player()] -= bet_amount
            elif action == Action.ALL_IN:
                self._current_bet = self.players_stack[self.current_player()]
                self.pot[self.current_player()] = self._current_bet
                self.players_stack[self.current_player()] = 0

            if self._is_betting_round_complete():
                self._advance_round()
                self._current_bet = 0
                self._next_player = min(self._active_players)
                if self._should_end_game():
                    self._game_over = True
                    for i in range(_NUM_PLAYERS):
                        self.players_stack[i] += self.returns()[i]
                else:
                    self._advance_to_next_player()
            else:
                self._advance_to_next_player()
            
            
    def _advance_round(self):
        """Advances the game to the next round (e.g., preflop → flop)."""
        if self._round == "preflop":
            self._round = "flop"
        elif self._round == "flop":
            self._round = "turn"
        elif self._round == "turn":
            self._round = "river"
        else:
            self._round = "showdown"

        # Reset betting state for the new round
        self._current_bet = 0
        for i in range(_NUM_PLAYERS):
            self.cumulative_pot[i] += self.pot[i]
        self.pot = [0.0] * _NUM_PLAYERS
    def _should_end_game(self):
        """
    Determines if the game should end based on:
    - Minimum pot contributions.
    - Number of actions taken.
    """
        return (len(self._active_players)==1 or self._round == "showdown")

    def _action_to_string(self, player, action):
        """Converts an action to a human-readable string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{action}"  # For chance actions (dealing cards)
        elif action == Action.POST_SB:
            return "Post SB"
        elif action == Action.POST_BB:
            return "Post BB"
        elif action == Action.FOLD:
            return "Fold"
        elif action == Action.CALL:
            return "Call"
        elif action == Action.BET_1_5:
            return "Bet 1.5 BB"
        elif action == Action.BET_3:
            return "Bet 3 BB"
        elif action == Action.BET_5:
            return "Bet 5 BB"
        elif action == Action.RAISE_2_5:
            return "Raise 2.5 BB"
        elif action == Action.RAISE_5:
            return "Raise 5 BB"
        elif action == Action.RAISE_8:
            return "Raise 8 BB"
        elif action == Action.ALL_IN:
            return "All-In"
        else:
            return "Unknown Action"

    def is_terminal(self):
        """Returns True if the game is over."""
        # The game ends if `_game_over` is True or other custom conditions
        return self._game_over

    def returns(self):
        """Calculate the total reward for each player at the end of the game."""
        if not self._game_over:
            return [0.0] * _NUM_PLAYERS
        else:
            #Total pot size
            winnings = sum(self.cumulative_pot)
            if len(self._active_players) == 1 : 
                returns = [0.0] * _NUM_PLAYERS
                remaining_player, = self._active_players
                returns[remaining_player] = winnings
                return returns
            else:

                # Evaluate each player's best hand
                players_best_hands = []
                for i in range(_NUM_PLAYERS):
                    if i in self._active_players:
                        hole_cards = self.cards[i * 2:(i + 1) * 2]  # Get player's hole cards
                        best_hand = evaluator.evaluate_hand(hole_cards, self._community_cards)
                        players_best_hands.append(best_hand)

                # Determine the winning hand(s)
                max_hand = max(players_best_hands)
                winners = [i for i, hand in enumerate(players_best_hands) if hand == max_hand]

                # Distribute the pot among winners
                reward = winnings / len(winners) if winners else 0.0

                # Calculate rewards for each player
                rewards = []
                for i in range(_NUM_PLAYERS):
                    if i in winners:
                        rewards.append(reward)  # Winners get a share of the pot
                    else:
                        rewards.append(-self.pot[i])  # Losers lose their contribution to the pot

                return rewards
    def __str__(self):
        """String representation of the game state."""
        # Convert stored card tuples to strings (e.g., "2h")
        card_str = " ".join([f"{rank}{suit}" for (rank, suit) in self.cards])
        bet_str = " | ".join([
            self._action_to_string(i % _NUM_PLAYERS, action)
            for i, action in enumerate(self.bets)
        ])
        pot_str = ", ".join([f"Player {i}: {amt} BB" for i, amt in enumerate(self.pot)])
        current_player = f"Current Player: {self.current_player()}"
        game_status = "Game Over" if self._game_over else "In Progress"
        
        return (f"Cards: {card_str}\n"
                f"Bets: {bet_str}\n"
                f"Pot: {pot_str}\n"
                f"{current_player}\n"
                f"Status: {game_status}")
class KickOffObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(
                f"Observation parameters not supported; passed {params}")

        # Determine which observation pieces we want to include.
        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS, ))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("private_card", len(_DECK), (len(_DECK), )))
        if iig_obs_type.public_info:
            if iig_obs_type.perfect_recall:
                pieces.append(
                    ("betting", len(Action), (_NUM_PLAYERS, len(Action))))
            else:
                pieces.append(
                    ("pot_contribution", _NUM_PLAYERS, (_NUM_PLAYERS, )))

        # Build the single flat tensor.
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        # Build the named & reshaped views of the bits of the flat tensor.
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        if "player" in self.dict:
            # Only set if valid player
            if 0 <= player < _NUM_PLAYERS:
                if "player" in self.dict:
                    self.dict["player"][player] = 1
        # Handle private card only for valid players and when cards are dealt
        if "private_card" in self.dict:
            if 0 <= player < _NUM_PLAYERS and len(state.cards) > player:
                card = state.cards[player]
                self.dict["private_card"][tuple(_DECK).index(card)] = 1  # Use tuple for indexing
        if "pot_contribution" in self.dict:
            self.dict["pot_contribution"][:] = state.pot
        if "betting" in self.dict:
            for turn, action in enumerate(state.bets):
                self.dict["betting"][turn % _NUM_PLAYERS, action] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")
        if "private_card" in self.dict and len(state.cards) > player:
            pieces.append(f"card:{state.cards[player]}")
        if "pot_contribution" in self.dict:
            pot_str = " ".join([f"{amt} BB" for amt in state.pot])
            pieces.append(f"pot[{pot_str}]")
        if "betting" in self.dict and state.bets:
            action_str = " | ".join([
                self._action_to_string(turn % _NUM_PLAYERS, action)
                for turn, action in enumerate(state.bets)
            ])
            pieces.append(f"betting[{action_str}]")
        return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, KickOffGame)
