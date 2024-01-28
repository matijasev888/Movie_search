
import pandas as pd
import Levenshtein as lev
from collections import Counter

class WordMatcher:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.dictionary = self._create_dictionary()

    def _create_dictionary(self):
        custom_dictionary = set(self.df[self.df.columns[0]].tolist())
        return [str(word) for word in custom_dictionary if pd.notnull(word)]

    @staticmethod
    def _jaccard_similarity(str1, str2):
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    @staticmethod
    def _levenshtein_similarity(input_word, dict_word):
        lev_distance = lev.distance(input_word, dict_word)
        max_length = max(len(input_word), len(dict_word))
        return 1 - (lev_distance / max_length) if max_length != 0 else 0

    def _combined_similarity(self, input_word, dict_word):
        jaccard_sim = self._jaccard_similarity(input_word, dict_word)
        lev_similarity = self._levenshtein_similarity(input_word, dict_word)
        return (jaccard_sim + lev_similarity) / 2

    def find_close_matches(self, word, similarity_threshold=0.8, num_matches=7):
        similarities = []
        for dict_word in self.dictionary:
            similarity = self._combined_similarity(word, dict_word)
            if similarity >= similarity_threshold:
                similarities.append((similarity, dict_word))
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [word for _, word in similarities[:num_matches]]
