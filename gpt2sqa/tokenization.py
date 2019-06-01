# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import regex as re
import json
import sys
import logging
import os
import unicodedata
from io import open

from .file_utils import cached_path

logger = logging.getLogger(__name__)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",}
PRETRAINED_MERGES_ARCHIVE_MAP = {'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {'gpt2': 1024,}
VOCAB_NAME = 'vocab.json'
MERGES_NAME = 'merges.txt'
SPECIAL_TOKENS_NAME = 'special_tokens.txt'


@lru_cache()
def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer(object):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level BPE
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path='gpt2', cache_dir=None, *inputs, **kwargs):
        """
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            merges_file = PRETRAINED_MERGES_ARCHIVE_MAP[pretrained_model_name_or_path]
            special_tokens_file = None
        else:
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_NAME)
            merges_file = os.path.join(pretrained_model_name_or_path, MERGES_NAME)
            special_tokens_file = os.path.join(pretrained_model_name_or_path, SPECIAL_TOKENS_NAME)
            if not os.path.exists(special_tokens_file):
                special_tokens_file = None
            else:
                logger.info("loading special tokens file {}".format(special_tokens_file))
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
            resolved_merges_file = cached_path(merges_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    pretrained_model_name_or_path,
                    vocab_file, merges_file))
            return None
        if resolved_vocab_file == vocab_file and resolved_merges_file == merges_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
            logger.info("loading merges file {}".format(merges_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
            logger.info("loading merges file {} from cache at {}".format(
                merges_file, resolved_merges_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        if special_tokens_file and 'special_tokens' not in kwargs:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]
        else:
            special_tokens = kwargs.pop('special_tokens', [])
        tokenizer = cls(resolved_vocab_file, resolved_merges_file, special_tokens=special_tokens, *inputs, **kwargs)
        return tokenizer

    def __init__(self, vocab_file, merges_file, errors='replace', special_tokens=None, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
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
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        merge_file = os.path.join(vocab_path, MERGES_NAME)
        special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        index = len(self.encoder)
        with open(special_tokens_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving special tokens vocabulary to {}: BPE indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(special_tokens_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        return vocab_file, merge_file, special_tokens_file


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
