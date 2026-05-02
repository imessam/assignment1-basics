import os
import sys
import regex as re
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

    def __init__(self):

        self._tok_EOS = "<|endoftext|>"
        self._specials = {0 : self._tok_EOS}

        self._vocab = [self._tok_EOS] +  [chr(i) for i in range(255)]
        
        self._str_to_idx = {s : idx for (idx, s) in enumerate(self._vocab)}
        self._idx_to_str = {idx : s for (idx, s) in enumerate(self._vocab)}
        self._idx_to_byte = {idx : bytes(s, "utf-8") for (idx, s) in enumerate(self._vocab)}

    def _pretokenize(self, corpus : List[str]) -> List[str]:

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = []

        for sentence in corpus:
            pre_tokens.extend(re.findall(PAT, sentence))

        return pre_tokens

    def _count(self, pre_tokens : List[str]) -> Dict[str, int]:
        
        words_count = {}

        for token in pre_tokens:
            token_stripped = token.strip()
            if token_stripped not in words_count:
                words_count[token_stripped] = 0
            words_count[token_stripped] += 1

        return words_count
    
    def _counts_to_bytes(self, words_count : Dict[str, int]) -> Dict[Tuple[bytes, ...], int]:

        bytes_count : dict[tuple[bytes, ...], int] = {}

        for word, count in words_count.items():
            bytes_count[tuple([ch.encode() for ch in word])] = count

        return bytes_count
    

    def _extract_pairs(self, bytes_count : Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:

        pairs = {}

        for token_bytes, count in bytes_count.items():

            for idx, b in enumerate(token_bytes):

                if idx > 0 and idx < len(token_bytes):

                    pair = (token_bytes[idx-1] , token_bytes[idx])

                    if pair not in pairs:
                        pairs[pair] = 0

                    pairs[pair] += count

        return pairs
    
    def _extract_max_pair(self, pairs : Dict[Tuple[bytes,bytes], int]) -> Tuple[Tuple[bytes,bytes], int]:

        max_pair : Tuple = ()
        max_count = 0

        for pair, count in pairs.items():
            if count > max_count:
                max_pair = pair
                max_count = count
            elif count == max_count:
                pair_str = pair[0] + pair[1]
                max_pair_str = max_pair[0] + max_pair[1]
                max_pair = pair if pair_str > max_pair_str else max_pair
        
        return max_pair, max_count


    def _merge(self, pairs : Dict[Tuple[bytes, bytes], int],  bytes_count : Dict[Tuple[bytes, ...], int],  merges : List[Tuple[bytes, bytes]]) -> Dict[Tuple[bytes, ...], int]:
        
        bytes_count_merged : Dict[Tuple[bytes, ...], int] = {}

        max_pair, pair_count = self._extract_max_pair(pairs)
        # print(f"Max pair : {max_pair}, pair count : {pair_count}")

        if pair_count == 0:
            return bytes_count_merged

        merges.append(max_pair)

        max_pair_str = max_pair[0] + max_pair[1]

        for token_bytes, count in bytes_count.items():

            new_token_bytes = []

            num_tokens = len(token_bytes)
            idx = 0

            while idx < num_tokens:

                pair = None

                if (idx < len(token_bytes) -1):
                    pair = token_bytes[idx] + token_bytes[idx+1]

                if pair and pair == max_pair_str:
                    new_token_bytes.append(pair)
                    idx += 1
                else : 
                    new_token_bytes.append(token_bytes[idx])

                idx += 1
            
            bytes_count_merged[tuple(new_token_bytes)] = count


        return  bytes_count_merged
    
    def _update_vocab(self, merges : List[Tuple[bytes, bytes]]) -> Dict[int, bytes]:

        for merge in merges:
        
            merged = merge[0] + merge[1]
            merged_str = merged.decode()

            self._vocab.append(merged_str)

            idx = len(self._vocab)

            if merged_str not in self._str_to_idx:
            
                self._str_to_idx[merged_str] = idx
                self._idx_to_str[idx] =   merged_str
                self._idx_to_byte[idx] = merged     

        return self._idx_to_byte     

    def train(self, corpus : List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        pre_tokens : List = self._pretokenize(corpus = corpus)
        # print(f"Pre-tokens : {pre_tokens} ")

        counts : Dict[str, int] = self._count(pre_tokens)
        # print(f"Counts : {counts}")

        bytes_count = self._counts_to_bytes(counts)
        # print(f"Bytes Count : {bytes_count}")

        merges : List[Tuple[bytes, bytes]] = []

        old_size = -1
        new_size = len(merges)

        while old_size != new_size:
            
            old_size = new_size

            pairs = self._extract_pairs(bytes_count)
            # print(f"Pairs : {pairs}")

            bytes_count_merged = self._merge(pairs, bytes_count, merges)
            # print(f"merges : {merges}")

            bytes_count = bytes_count_merged

            new_size = len(merges)

        
        self._idx_to_byte = self._update_vocab(merges)
        # print(f"vocab : {self._idx_to_byte}")


        return self._idx_to_byte, merges
    

def train_bpe(input_path : str, vocab_size : int, special_tokens : List[str]) -> Tuple[dict[int, bytes], List[Tuple[bytes, bytes]]]:

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
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print(f"boundaries : {boundaries}")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_wo_spc_tok = re.split(re.escape("|".join(special_tokens)), chunk)
            
            vocab, merges = bpe.train(chunk_wo_spc_tok)

            break
        
        print(f"vocab : {vocab}")
            
    return vocab, merges



if __name__ == "__main__":

    input_path = sys.argv[1]

    train_bpe(input_path, 16000, ["<|endoftext|>"])

    # bpe = BPE()

    # bpe.train(examples)