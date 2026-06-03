import sys
from typing import List
from bpe import BPE

def encode(sentences : List[str]):

    bpe = BPE()

    bpe.encode_iterbale(sentences)


if __name__ == "__main__":

    vocab_path = sys.argv[1]
    merges_path = sys.argv[2]

    BPE.from_files(vocab_path, merges_path)

    examples = ["the cat ate"]

    encode(examples)
    