import os
import json
import nnabla_func as nf
from functools import lru_cache

class Tokenizer():
    def __init__(self, hfm):
        with open(os.path.join(hfm.path, hfm.files.vocab), "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(os.path.join(hfm.path, hfm.files.merges), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if(i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

    def __call__(self, text):
        bpe_tokens = self.bpe(text).split(" ")
        token_ids = [self.encoder[token] for token in bpe_tokens if token in self.encoder]
        return token_ids
    @lru_cache
    def bytes_to_unicode(self):
        bs = (
            list(range(ord("!"), ord("~") + 1 )) + list(range(ord("¡"), ord("~") + 1)) + list(range(ord("@"), ord("y") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

class Tinyswallow_model():
    def __init__(self, huggingface_model):
        self.all_token_ids = []
        self.state_dict = huggingface_model.state_dict

        self.config = None
        self.cache = None
        self.config_data = huggingface_model.config_data
        self.bos_token = self.tokenizer.encoder.get("<BOS>")
        self.eos_token = self.tokenizer.encoder.get("<EOS>")

        self.tokenizer = Tokenizer(huggingface_model)
        print(self.tokenizer("hello world"))

    def add_text(self, text):
        token_ids = self.tokenizer(text)
        self.all_token_ids.append(token_ids)
        x = nf.embed_tokens(token_ids)
        x = nf.rotary_emb(x)
        self.decoder_layer.generate_kv(x)

    def remove_text(self, c: int = 1):
        self.all_token_ids = self.all_token_ids[:-c]
        cache_data.remove_kv(c)

    def generate(self, max_length: int = 1):
        token_ids = [self.bos_token]
        x = [self.bos_token]
        for _ in range(max_length):
            x = nf.embed_tokens(x)
            x = nf.decoder_layer(self.config_data.max_layer, x)
            if last_token == self.eos_token:
                break
            x = nf.rms_norm(state_dict[f"model.norm.weight"], x)
            logits = nf.im_head(x)
            last_token = int(logits[0].argmax())
            token_ids.append(last_token)
            x = [last_token]
        self.all_token_ids.append(token_ids)
        tokens = [self.tokenizer.decoder[t] for t in token_ids]
        text_out = "".join(tokens)
        return text_out
