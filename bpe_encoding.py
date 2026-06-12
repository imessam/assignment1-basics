import sys
from typing import Iterator, List
from bpe import BPE

def encode(bpe : BPE, sentences : List[str]) -> Iterator[int]:

    return bpe.encode_iterbale(sentences)


if __name__ == "__main__":

    vocab_path = sys.argv[1]
    merges_path = sys.argv[2]

    bpe = BPE.from_files(vocab_path, merges_path)

    examples = ["the cat ate", "the dog sleep"]

    encodings = encode(bpe, examples)
    print(f"encodings : {list(encodings)}")
    