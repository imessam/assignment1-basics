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

    tokenizer = BPE.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    examples = ["Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"]

    # encodings = encode(tokenizer, examples)
    # encodings_list = list(encodings)

    # print(f"encodings : {encodings_list}")

    # decoded_str = decode(tokenizer, encodings_list)
    # print(f"decoded_str : {decoded_str}")

    encoded_ids = tokenizer.encode(examples[0])

    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]

    print(tokenizer.decode(encoded_ids), tokenized_string, tokenized_string.count("<|endoftext|>"))
    