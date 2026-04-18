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

    def _pretokenize(self, corpus : List[str]) -> List[str]:

        pre_tokens = []

        for sentence in corpus:
            pre_tokens.extend(sentence.split(" "))

        return pre_tokens

    def _count(self, pre_tokens : List[str]) -> Dict[str, int]:
        
        words_count = {}

        for token in pre_tokens:
            if token not in words_count:
                words_count[token] = 0
            words_count[token] += 1

        return words_count
    
    def _counts_to_bytes(self, words_count : Dict[str, int]) -> Dict[Tuple[bytes, ...], int]:

        bytes_count : dict[tuple[bytes, ...], int] = {}

        for word, count in words_count.items():
            bytes_count[tuple([ch.encode() for ch in word])] = count

        return bytes_count


    def _merge(self, bytes_count : Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
        
        bytes_count_merged : Dict[Tuple[bytes, ...], int] = {}

        pairs = {}

        for token_bytes, count in bytes_count.items():

            for idx, b in enumerate(token_bytes):

                if idx > 0 and idx < len(token_bytes):

                    pair = token_bytes[idx-1] + token_bytes[idx]
                    
                    if pair not in pairs:
                        pairs[pair] = 0

                    pairs[pair] += count


        print(f"Pairs : {pairs}")

        return bytes_count_merged

    def train(self, corpus : List[str]) -> Dict:

        vocab = {}

        pre_tokens : List = self._pretokenize(corpus = corpus)
        print(f"Pre-tokens : {pre_tokens} ")

        counts : Dict[str, int] = self._count(pre_tokens)
        print(f"Counts : {counts}")

        bytes_count = self._counts_to_bytes(counts)
        print(f"Bytes Count : {bytes_count}")

        bytes_count_merged = self._merge(bytes_count)

        return vocab


if __name__ == "__main__":

    bpe = BPE()

    bpe.train(examples)

    
    