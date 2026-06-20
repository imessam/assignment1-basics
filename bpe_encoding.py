import sys

from typing import Iterator, List
from bpe import BPE

def encode(bpe : BPE, sentences : List[str]) -> Iterator[int]:

    return bpe.encode_iterable(sentences)

def decode(bpe : BPE, tokens_ids : List[int]) -> str:

    return bpe.decode(tokens_ids)


if __name__ == "__main__":

    vocab_path = sys.argv[1]
    merges_path = sys.argv[2]

    tokenizer = BPE.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])

    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    print(tokenized_string, tokenizer.decode(ids))
    # Test roundtrip

    