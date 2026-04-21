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
    

    def _extract_pairs(self, bytes_count : Dict[Tuple[bytes, ...], int]) -> Dict[bytes, int]:

        pairs = {}

        for token_bytes, count in bytes_count.items():

            for idx, b in enumerate(token_bytes):

                if idx > 0 and idx < len(token_bytes):

                    pair = token_bytes[idx-1] + token_bytes[idx]

                    if pair not in pairs:
                        pairs[pair] = 0

                    pairs[pair] += count

        return pairs
    
    def _extract_max_pair(self, pairs : Dict[bytes, int]) -> Tuple[bytes, int]:

        max_pair_idx = b""
        max_count = 0

        for pair, count in pairs.items():
            if count > max_count:
                max_pair_idx = pair
                max_count = count
            elif count == max_count:
                max_pair_idx = pair if pair > max_pair_idx else max_pair_idx
        
        
        return max_pair_idx, max_count


    def _merge(self, pairs : Dict[bytes, int],  bytes_count : Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
        
        bytes_count_merged : Dict[Tuple[bytes, ...], int] = {}

        max_pair, _ = self._extract_max_pair(pairs)

        for token_bytes, count in bytes_count.items():

            new_token_bytes = []

            for idx, b in enumerate(token_bytes):

                if idx < len(token_bytes)-1:

                    pair = token_bytes[idx] + token_bytes[idx+1]

                    if pair == max_pair:
                        new_token_bytes.append(pair)
                    elif idx == len(token_bytes)-2:
                        new_token_bytes.append(token_bytes[idx])
                        new_token_bytes.append(token_bytes[idx+1])
                    else : 
                        new_token_bytes.append(token_bytes[idx])
            
            bytes_count_merged[tuple(new_token_bytes)] = count


        return  bytes_count_merged

    def train(self, corpus : List[str]) -> Dict:

        vocab = {}

        pre_tokens : List = self._pretokenize(corpus = corpus)
        print(f"Pre-tokens : {pre_tokens} ")

        counts : Dict[str, int] = self._count(pre_tokens)
        print(f"Counts : {counts}")

        bytes_count = self._counts_to_bytes(counts)
        print(f"Bytes Count : {bytes_count}")

        for _ in range(6):

            pairs = self._extract_pairs(bytes_count)
            print(f"Pairs : {pairs}")

            bytes_count_merged = self._merge(pairs, bytes_count)
            print(f"Merged : {bytes_count_merged}")

            bytes_count = bytes_count_merged


        return vocab


if __name__ == "__main__":

    bpe = BPE()

    bpe.train(examples)

    
    