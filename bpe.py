import os
import time
import pickle

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
        self._special_tokens_bytes : List[bytes] = []
        

        self._merges : List[Tuple[bytes, bytes]] = []
        self._idx_to_byte : Dict[int, bytes] = {}
        self._byte_to_idx : Dict[bytes, int] = {}

        if special_tokens:
            
            for spc_token in special_tokens:
                if spc_token not in self._special_tokens:
                    self._special_tokens.append(spc_token)

            self._special_tokens_bytes = [special_token.encode() for special_token in self._special_tokens]

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

            self._byte_to_idx = {b : idx for idx, b in self._idx_to_byte.items()} 

            count = len(vocab)

            for special_token in self._special_tokens_bytes:

                if special_token not in self._byte_to_idx:

                    self._idx_to_byte[count] = special_token
                    self._byte_to_idx[special_token] = count

                    count += 1
        

            # print(self._special_tokens, self._byte_to_idx)

    def _pretokenize_train(self, corpus : Iterable[str]) -> List[bytes]:

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens_bytes = []

        for sentence in tqdm(corpus, desc = "Pretokenizing ..."):
            
            for match in re.finditer(PAT, sentence):
                pre_tokens_bytes.append(match.group().encode())

        return pre_tokens_bytes

    def _pretokenize_encode(self, corpus : Iterable[str]) -> List[bytes]:

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens_bytes = []
        sentences_wo_spc_tok = []

        for sentence in tqdm(corpus, desc = "Pretokenizing ..."):
            
            sentences_wo_spc_tok = re.split(re.escape("|".join(self._special_tokens)), sentence)
            special_tokens_matches = re.findall(re.escape("|".join(self._special_tokens),), sentence)

            for sentence_wo_spc_tok  in sentences_wo_spc_tok :
                for match in re.finditer(PAT, sentence_wo_spc_tok):
                    pre_tokens_bytes.append(match.group().encode())

                if len(special_tokens_matches) > 0:
                    pre_tokens_bytes.append(special_tokens_matches.pop(0).encode())

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
            if pretoken in self._special_tokens_bytes:
                pretokens_bytes.append(tuple([pretoken]))
            else:
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
    
    def _merge_encodings(self, encodings : List[Tuple[bytes, ...]]) -> List[Tuple[bytes, ...]]:

        encodings_old = encodings.copy()
        encodings_merged : List[Tuple[bytes, ...]] = []

        is_merged = True

        while is_merged:

            is_merged = False
            encodings_merged.clear()
            print(encodings_old)

            for pretoken in encodings_old:

                new_pretoken = []
                num_bytes = len(pretoken)

                idx = 0

                while idx < num_bytes:

                    pair_byte = None

                    if idx < (len(pretoken) - 1) :

                        pair_byte = pretoken[idx] + pretoken[idx + 1]

                        print(pair_byte)
                    if pair_byte and pair_byte in self._byte_to_idx:
                            print("found")
                            idx += 1
                            is_merged = True
                    else:
                        new_pretoken.append(pretoken[idx])

                    idx += 1

                encodings_merged.append(tuple(new_pretoken))
            
            encodings_old = encodings_merged.copy()


        return encodings_merged
    
    @classmethod
    def from_files(cls, vocab_filepath : str, merges_filepath : str , special_tokens : Union[None, List[str]] = None):

        if not os.path.exists(vocab_filepath) or not os.path.exists(merges_filepath):
            return BPE()
        
        vocab : Dict[int, bytes] = {}
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return BPE(vocab, merges, special_tokens)


    def pretokenize(self, corpus : List[str]) -> Counter:

        print(f"Pretokinizing ...")

        t1 = time.perf_counter()

        pre_tokens_bytes = self._pretokenize_train(corpus)
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

        return list(self.encode_iterable([text].__iter__()))


    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:

        print("Encoding ... ")

        encodings : List[int] = []

        pretokens = self._pretokenize_encode(iterable)
        # print(f"pretokens : {pretokens}")

        pretokens_bytes = self._pretokens_to_bytes(pretokens)
        # print(f"pretokens_bytes : {pretokens_bytes}")

        encodings_merged = self._merge_encodings(pretokens_bytes)
        # print(f"encodings_merged : {encodings_merged}")

        encodings = [self._byte_to_idx[b] for encoding in encodings_merged for b in encoding]

        return encodings.__iter__()
    
    def decode(self, ids: list[int]) -> str :

        tokens = b""

        for id in ids:
            byte_token = self._idx_to_byte.get(id)

            if byte_token is None:
                print(f"{id} not found ...")
                continue

            tokens += byte_token

        decoded_str = tokens.decode(errors="replace")
            
        return decoded_str



