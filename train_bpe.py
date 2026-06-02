import os
import sys
import time
import pickle
import shutil

import regex as re

from tqdm.auto import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
from typing import Dict, List, Tuple
from bpe import BPE

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

    output_path = os.path.join("output")

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

    out_dir = os.path.join(output_path, f"trained_{input_file_name}")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:

        pickle.dump(vocab, f)

    with open(os.path.join(out_dir, "merges.pkl"), "wb") as f:

        pickle.dump(merges, f)

    return vocab, merges



if __name__ == "__main__":

    input_path = sys.argv[1]
    max_vocab = int(sys.argv[2])

    vocab, merges = train_bpe(input_path, max_vocab, ["<|endoftext|>"])