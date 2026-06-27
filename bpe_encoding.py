import sys

from typing import Iterator, List
from bpe import BPE

def encode(bpe : BPE, sentences : List[str]) -> Iterator[int]:

    return bpe.encode_iterable(sentences)

def decode(bpe : BPE, tokens_ids : List[int]) -> str:

    return bpe.decode(tokens_ids)

import tiktoken

if __name__ == "__main__":

    vocab_path = sys.argv[1]
    merges_path = sys.argv[2]

    tokenizer = BPE.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    reference_tokenizer = tiktoken.get_encoding("gpt2")

    test_string = "government"

    ids = tokenizer.encode(test_string)
    ref_ids = reference_tokenizer.encode(test_string)

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    ref_tokenized_string = [reference_tokenizer.decode([x]) for x in ref_ids]
    print(ids, ref_ids, tokenized_string, ref_tokenized_string, tokenizer.decode(ids))
    