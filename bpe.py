import time

import regex as re

from tqdm.auto import tqdm
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Tuple, Union


class BPE:

    def __init__(self, 
                 vocab : Union[None, Dict[int, bytes]] = None, 
                 merges : Union[None, List[Tuple[bytes, bytes]]] = None, 
                 special_tokens: Union[list[str], None] = None,
                 vocab_size : int = 1000):

        self._vocab_size = vocab_size

        self._tok_EOS = "<|endoftext|>"
        self._special_tokens : List[str] = [self._tok_EOS]

        self._merges : List[Tuple[bytes, bytes]] = []
        self._idx_to_byte : Dict[int, bytes] = {}

        if special_tokens:
            self._special_tokens.extend(special_tokens)

        if (not vocab) or (not merges): 
            self._vocab : List[bytes] = [special_token.encode() for special_token in self._special_tokens] +  [bytes([i]) for i in range(256)] 
            self._vocab_idx = len(self._vocab)
            self._rem_vocab = vocab_size - len(self._vocab)
            self._vocab.extend([b""] * self._rem_vocab)

            self._merges : List[Tuple[bytes, bytes]] = [(b"", b"")] * (self._rem_vocab)
            self._merge_idx = 0

        else:
            self._idx_to_byte = vocab
            self._merges = merges


    def _pretokenize(self, corpus : List[str]) -> List[bytes]:

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens_bytes = []

        for sentence in tqdm(corpus, desc = "Pretokenizing ..."):
            for match in re.finditer(PAT, sentence):
                pre_tokens_bytes.append(match.group().encode())

        return pre_tokens_bytes

    def _count(self, pre_tokens_bytes : List[bytes]) -> Counter:
        
        words_bytes_count = Counter()

        for token in tqdm(pre_tokens_bytes, desc = "Counting ..."):
            words_bytes_count[token] += 1

        return words_bytes_count

    
    def _words_to_bytes(self, words_count : Counter[bytes]) -> Dict[Tuple[bytes, ...], int]:

        bytes_count : dict[tuple[bytes, ...], int] = {}

        for word, count in tqdm(words_count.items(), desc= "Word to bytes ..."):
            bytes_count[tuple([bytes([ch]) for ch in word])] = count

        return bytes_count
    
    def _pretokens_to_bytes(self, pretokens : List[bytes]) -> List[Tuple[bytes, ...]]:

        pretokens_bytes : List[tuple[bytes, ...]] = []

        for pretoken in tqdm(pretokens, desc= "pretokens to bytes ..."):
            pretokens_bytes.append(tuple([bytes([ch]) for ch in pretoken]))

        return pretokens_bytes
    

    def _extract_pairs(self, bytes_count : Dict[Tuple[bytes, ...], int]) -> Tuple[Tuple[bytes,bytes], int]:

        pairs : Counter[Tuple[bytes, bytes]] = Counter()
        max_pair : Tuple = ()
        max_count = 0

        for word_bytes, count in bytes_count.items():

            for idx in range(len(word_bytes) - 1):

                pair = (word_bytes[idx] , word_bytes[idx + 1])
    
                pairs[pair] += count

                curr_count = pairs[pair]

                if curr_count > max_count:
                    max_pair = pair
                    max_count = curr_count
                elif curr_count == max_count:
                    max_pair = max([pair, max_pair])
                    max_count = curr_count
                    
        
        return max_pair, max_count
    
    
    def _merge_enhanced(self, bytes_count : Dict[Tuple[bytes, ...], int]) -> None:
        
        while self._vocab_idx < self._vocab_size:

            max_pair, pair_count = self._extract_pairs(bytes_count)

            if pair_count == 0:
                break

            self._merges[self._merge_idx] = max_pair
            self._merge_idx += 1

            merged = max_pair[0] + max_pair[1]

            if self._vocab[self._vocab_idx] != merged:
                
                self._vocab[self._vocab_idx] = merged
                self._vocab_idx += 1
            
            word_bytes_list = list(bytes_count.keys())
            for word_bytes in word_bytes_list:

                count = bytes_count[word_bytes]

                new_word_bytes = []

                num_bytes = len(word_bytes)
                is_found = False
                i = 0

                while i < num_bytes:

                    pair = None

                    if i < (num_bytes - 1):
                        pair = word_bytes[i] + word_bytes[i+1]

                    if pair and pair == merged:
                        new_word_bytes.append(pair)
                        i += 1
                        is_found = True
                    else : 
                        new_word_bytes.append(word_bytes[i])

                    i += 1

                if is_found:
                    bytes_count.pop(word_bytes)
                    bytes_count[tuple(new_word_bytes)] = count
            
        return
    
    @classmethod
    def from_files(cls, vocab_filepath : str, merges_filepath : str , special_tokens : Union[None, List[str]] = None):

        return cls


        

    def pretokenize(self, corpus : List[str]) -> Counter:

        print(f"Pretokinizing ...")

        t1 = time.perf_counter()

        pre_tokens_bytes = self._pretokenize(corpus)
        words_bytes_count = self._count(pre_tokens_bytes)

        t2 = time.perf_counter()

        print(f"pretokenization done, elapsed : {(t2-t1) / 60.0} minutes")

        return words_bytes_count
    
    def train(self, words_bytes_count : Counter) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        bytes_count = self._words_to_bytes(words_bytes_count)

        t1 = time.perf_counter()

        self._merge_enhanced(bytes_count)
        self._idx_to_byte : Dict[int, bytes] = {idx : b for idx, b in enumerate(self._vocab)}

        t2 = time.perf_counter()

        print(f"Train done, elapsed : {(t2-t1) / 60.0} minutes")

        return self._idx_to_byte, self._merges
    
    def encode(self, text : str) -> List[int]:

        print("Encoding ... ")

        encodings = []


        return encodings


    def encode_iterbale(self, iterable : Iterable[str]) -> Iterator[int]:

        print("Encoding ... ")

        encodings : List[int] = []

        pretokens = self._pretokenize(iterable)
        print(f"pretokens : {pretokens}")

        pretokens_bytes = self._pretokens_to_bytes(pretokens)
        print(f"pretokens_bytes : {pretokens_bytes}")


        return encodings
    
    def decode(self, ids: list[int]) -> str :

        print("Decoding ...")

        decoded = ""

        return decoded



