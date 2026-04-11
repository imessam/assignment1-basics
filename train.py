from typing import Dict, List, Tuple

examples = [
"low low low low low",
"lower lower widest widest widest",
"newest newest newest newest newest newest"]


class BPE:

    def __init__(self):
        self._vocab = []
        self._tok_EOS = "<|endoftext|>"
        self._specials = {256 : self._tok_EOS}

    def pretokenize(self, corpus : List[str]) -> List:
        return [sentence.split(" ") for sentence in corpus]

    def count(self, pre_tokens : List[str]) -> Dict[str, int]:
        words_count = {}

        for token in pre_tokens:
            if token not in words_count:
                words_count[token] = 0
            words_count[token] += 1

        return words_count
    
    def count_to_bytes(self, words_count : Dict[str, int]) -> Dict[Tuple[bytes, ...], int]:

        bytes_count : dict[tuple[bytes, ...], int] = {}

        return bytes_count


    def merge(self):
        pass

    def train(self):
        pass


    