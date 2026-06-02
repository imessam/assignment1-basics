from typing import List
from bpe import BPE

def encode(sentences : List[str]):

    bpe = BPE()

    bpe.encode_iterbale(sentences)


if __name__ == "__main__":

    examples = ["the cat ate"]

    encode(examples)
    