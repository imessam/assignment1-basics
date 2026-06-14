import sys

from typing import Iterator, List
from bpe import BPE

def encode(bpe : BPE, sentences : List[str]) -> Iterator[int]:

    return bpe.encode_iterbale(sentences)

def decode(bpe : BPE, tokens_ids : List[int]) -> str:

    return bpe.decode(tokens_ids)


if __name__ == "__main__":

    vocab_path = sys.argv[1]
    merges_path = sys.argv[2]

    bpe = BPE.from_files(vocab_path, merges_path)

    examples = ["the cat ate , the dog sleep"]

    encodings = encode(bpe, examples)
    encodings_list = list(encodings)

    print(f"encodings : {encodings_list}")

    decoded_str = decode(bpe, encodings_list)
    print(f"decoded_str : {decoded_str}")
    