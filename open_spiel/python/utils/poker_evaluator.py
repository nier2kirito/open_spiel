import itertools

class PokerEvaluator:
    def __init__(self):
        self.rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                           '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    def evaluate_hand(self, hole_cards, community_cards):
        """Evaluate the best 5-card hand from 7 cards (2 hole + 5 community)"""
        all_cards = hole_cards + community_cards
        best_hand = None
        for combo in itertools.combinations(all_cards, 5):
            score = self._score_hand(combo)
            if (best_hand is None) or (score > best_hand):
                best_hand = score
        return best_hand

    def _score_hand(self, hand):
        """Return a tuple representing the hand strength for comparison"""
        ranks = sorted([self.rank_values[card[0]] for card in hand], reverse=True)
        suits = [card[1] for card in hand]
        rank_counts = self._count_ranks(ranks)
        flush = self._is_flush(suits)
        straight = self._is_straight(ranks)
        
        # Evaluate hand type
        if straight and flush:
            return (8, straight)
        if self._is_x_of_a_kind(4, rank_counts):
            return (7, self._get_x_rank(4, rank_counts), self._get_x_rank(1, rank_counts))
        if self._is_full_house(rank_counts):
            return (6, self._get_x_rank(3, rank_counts), self._get_x_rank(2, rank_counts))
        if flush:
            return (5, ranks)
        if straight:
            return (4, straight)
        if self._is_x_of_a_kind(3, rank_counts):
            kickers = self._get_kickers(ranks, [self._get_x_rank(3, rank_counts)])
            return (3, self._get_x_rank(3, rank_counts), kickers)
        if self._is_two_pair(rank_counts):
            pairs = sorted([k for k, v in rank_counts.items() if v == 2], reverse=True)[:2]
            kicker = self._get_kickers(ranks, pairs)
            return (2, pairs[0], pairs[1], kicker)
        if self._is_x_of_a_kind(2, rank_counts):
            pair_rank = self._get_x_rank(2, rank_counts)
            kickers = self._get_kickers(ranks, [pair_rank])
            return (1, pair_rank, kickers)
        return (0, ranks)

    def _count_ranks(self, ranks):
        return {rank: ranks.count(rank) for rank in set(ranks)}

    def _is_flush(self, suits):
        return len(set(suits)) == 1

    def _is_straight(self, ranks):
        unique_ranks = list(sorted(set(ranks), reverse=True))
        if len(unique_ranks) < 5:
            return False
            
        # Check for Ace-low straight (A-2-3-4-5)
        if unique_ranks == [14, 5, 4, 3, 2]:
            return 5
            
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                return unique_ranks[i]
        return False

    def _is_x_of_a_kind(self, x, rank_counts):
        return x in rank_counts.values()

    def _is_full_house(self, rank_counts):
        return (3 in rank_counts.values()) and (2 in rank_counts.values())

    def _is_two_pair(self, rank_counts):
        return list(rank_counts.values()).count(2) >= 2

    def _get_x_rank(self, x, rank_counts):
        for rank, count in rank_counts.items():
            if count == x:
                return rank
        return None

    def _get_kickers(self, all_ranks, exclude_ranks):
        return sorted([r for r in all_ranks if r not in exclude_ranks], reverse=True)[:5-len(exclude_ranks)]

    def compare_hands(self, hands):
        """Compare multiple hands and return the winning index(es)"""
        scored_hands = [self.evaluate_hand(h, []) for h in hands]
        max_score = max(scored_hands)
        return [i for i, score in enumerate(scored_hands) if score == max_score]
