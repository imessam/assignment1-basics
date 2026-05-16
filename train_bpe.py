import os
import sys
import time
import regex as re
from typing import BinaryIO, Union
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

    def __init__(self):

        self._tok_EOS = "<|endoftext|>"
        self._specials = {0 : self._tok_EOS}

        self._vocab : List[bytes] = [self._tok_EOS.encode()] +  [bytes([i]) for i in range(256)]
        self._merges : List[Tuple[bytes, bytes]] = []

        self._merge_idx = 0

        self._byte_to_idx = {b : idx for (idx, b) in enumerate(self._vocab)}
        self._idx_to_byte : Dict[int, bytes] = {idx : b for (idx, b) in enumerate(self._vocab)}

    def _pretokenize(self, corpus : List[str]) -> List[str]:

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = []

        for sentence in corpus:
            for match in re.finditer(PAT, sentence):
                pre_tokens.append(match.group())

        return pre_tokens

    def _count(self, pre_tokens : List[str]) -> Dict[str, int]:
        
        words_count = {}

        for token in pre_tokens:
            token_stripped = token
            if token_stripped not in words_count:
                words_count[token_stripped] = 0
            words_count[token_stripped] += 1

        return words_count
    
    def _words_to_bytes(self, words_count : Dict[str, int]) -> Dict[Tuple[bytes, ...], int]:

        bytes_count : dict[tuple[bytes, ...], int] = {}

        for word, count in words_count.items():
            bytes_count[tuple([ch.encode() for ch in word])] = count

        return bytes_count
    

    def _extract_pairs(self, bytes_count : Dict[Tuple[bytes, ...], int]) -> Tuple[Tuple[bytes,bytes], int]:

        pairs : Dict[bytes, int] = {}
        max_pair : Tuple = ()
        max_count = 0

        for word_bytes, count in bytes_count.items():

            for idx in range(len(word_bytes) - 1):

                pair = (word_bytes[idx] , word_bytes[idx + 1])
                pair_conc = pair[0] + pair[1]

                if pair_conc not in pairs:
                    pairs[pair_conc] = 0

                pairs[pair_conc] += count

                curr_count = pairs[pair_conc]

                if curr_count > max_count:
                    max_pair = pair
                    max_count = curr_count
                elif curr_count == max_count:
                    print(f"max [{pair} , {max_pair}] ")
                    max_pair = max([pair, max_pair])
                    max_count = curr_count
                    
        
        print(f"idx : {self._merge_idx} , Pairs : {pairs} , max_pair : {max_pair}, max_count : {max_count}")

        return max_pair, max_count
    


    def _merge(self,  max_pair : Tuple[bytes,bytes] , pair_count : int ,  bytes_count : Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
        
        bytes_count_merged : Dict[Tuple[bytes, ...], int] = {}

        if pair_count == 0:
            return bytes_count

        self._merges.append(max_pair)

        merged = max_pair[0] + max_pair[1]

        if merged not in self._byte_to_idx:
            idx = len(self._idx_to_byte)
            
            self._byte_to_idx[merged] = idx
            self._idx_to_byte[idx] = merged

        for word_bytes, count in bytes_count.items():

            new_word_bytes = []

            num_bytes = len(word_bytes)
            idx = 0

            while idx < num_bytes:

                pair = None

                if idx < (num_bytes - 1):
                    pair = word_bytes[idx] + word_bytes[idx+1]

                if pair and pair == merged:
                    new_word_bytes.append(pair)
                    idx += 1
                else : 
                    new_word_bytes.append(word_bytes[idx])

                idx += 1
            
            bytes_count_merged[tuple(new_word_bytes)] = count


        return  bytes_count_merged
       


    
    def train(self, corpus : List[str], vocab_size : int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        t1 = time.time()
        pre_tokens : List = self._pretokenize(corpus = corpus)
        t2 = time.time()
        print(f"Pre-tokens {pre_tokens} : , elapsed : {t2-t1} ")


        t1 = time.time()
        counts : Dict[str, int] = self._count(pre_tokens)
        t2 = time.time()
        print(f"Counts :  {counts} , elapsed : {t2-t1}")

        t1 = time.time()
        bytes_count = self._words_to_bytes(counts)
        t2 = time.time()
        print(f"Bytes Count : {bytes_count}  , elapsed : {t2-t1}")

        old_size = -1
        new_size = len(self._byte_to_idx)

        t1 = time.time()
        while (old_size != new_size) and (new_size < vocab_size):
            
            old_size = new_size

            t1_inner = time.time()
            max_pair, pair_count = self._extract_pairs(bytes_count)
            t2_inner = time.time()
            print(f"max_pair : {max_pair} , pair_count : {pair_count}  , elapsed : {t2_inner-t1_inner}")


            if pair_count == 0:
                break

            t1_inner = time.time()
            bytes_count_merged = self._merge(max_pair, pair_count , bytes_count)
            t2_inner = time.time()
            print(f"bytes_count_merged :  {bytes_count_merged} , elapsed : {t2_inner-t1_inner}")

            bytes_count = bytes_count_merged

            new_size = len(self._byte_to_idx)
            self._merge_idx += 1

        t2 = time.time()
        print(f", elapsed : {t2-t1}")

        # print(f"vocab {self._idx_to_byte} ")

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

    bpe = BPE()

    vocab : Dict[int, bytes] = {}
    merges : list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_wo_spc_tok = re.split(re.escape("|".join(special_tokens)), chunk)
            
            vocab, merges = bpe.train(chunk_wo_spc_tok, vocab_size)

        # print(f"vocab : {vocab}")
            
    return vocab, merges



if __name__ == "__main__":

    input_path = sys.argv[1]

    # vocab, merges = train_bpe(input_path, 1000, ["<|endoftext|>"])

    bpe = BPE()

    vocab, merges = bpe.train(examples, 160000)

    print(f"vocab : {vocab}")
    print(f"merges : {merges}")