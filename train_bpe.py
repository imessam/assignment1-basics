import os
import sys
import time
import datetime
import pickle

import regex as re

from tqdm.auto import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
from typing import Dict, List, Tuple

examples = [
"low low low low low",
"lower lower widest widest widest",
"newest newest newest newest newest newest"]




def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



class BPE:

    def __init__(self, vocab_size : int):

        self._vocab_size = vocab_size

        self._tok_EOS = "<|endoftext|>"
        self._specials = {0 : self._tok_EOS}

        self._vocab : List[bytes] = [self._tok_EOS.encode()] +  [bytes([i]) for i in range(256)] 
        self._vocab_idx = len(self._vocab)
        self._rem_vocab = vocab_size - len(self._vocab)
        self._vocab.extend([b""] * self._rem_vocab)

        self._merges : List[Tuple[bytes, bytes]] = [(b"", b"")] * (self._rem_vocab)
        self._merge_idx = 0

        self._idx_to_byte : Dict[int, bytes] = {}

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
    

def train_bpe(input_path : str | os.PathLike, vocab_size : int, special_tokens : List[str]) -> Tuple[dict[int, bytes], List[Tuple[bytes, bytes]]]:

    '''
        Args:

            input_path (str) :  Path to a text file with BPE tokenizer training data.
            
            vocab_size (int) : A positive integer that defines the maximum final vocabulary size (including
            the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
           
            special_tokens (list[str]) :  A list of strings to add to the vocabulary. During training, treat
            them as hard boundaries that prevent merges across their spans, but do not include them when
            computing merge statistics.
            Your BPE training function should return the resulting vocabulary and merges:
       
         Returns:

            vocab (dict[int, bytes]) : The tokenizer vocabulary, a mapping from int (token ID in the
            vocabulary) to bytes (token bytes).

            merges (list[tuple[bytes, bytes]]) : A list of BPE merges produced from training. Each list
            item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    '''

    output_path = os.path.join("output","trained_vocab")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bpe = BPE(vocab_size=vocab_size)

    vocab : Dict[int, bytes] = {}
    merges : list[tuple[bytes, bytes]] = []

    t1 = time.perf_counter()

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        all_chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunk_wo_spc_tok = re.split(re.escape("|".join(special_tokens)), chunk)

                all_chunks.append(chunk_wo_spc_tok)
        
        words_bytes_count_all = Counter()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:

            results = executor.map(bpe.pretokenize, all_chunks)

            for result in results:
                words_bytes_count_all.update(result)

    t2 = time.perf_counter()

    print(f"pretokenization the whole corpus done, elapsed : {(t2-t1) / 60.0} minutes")

    vocab, merges = bpe.train(words_bytes_count_all)

    input_file_name = ""
    if type(input_path) is str:
        input_file_name = input_path.split("/")[-1].split(".")[0]

    curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    out_file_name = f"trained_{input_file_name}_{curr_time}.pkl"

    with open(os.path.join(output_path, out_file_name), "wb") as f:

        pickle.dump(vocab, f)

    return vocab, merges



if __name__ == "__main__":

    input_path = sys.argv[1]
    max_vocab = int(sys.argv[2])

    vocab, merges = train_bpe(input_path, max_vocab, ["<|endoftext|>"])