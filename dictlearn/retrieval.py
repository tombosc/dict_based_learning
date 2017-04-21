"""Retrieve the dictionary definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
import shutil
import traceback
from collections import Counter, defaultdict

import numpy
import json
import nltk
from wordnik import swagger, WordApi, AccountApi

from dictlearn.util import vec2str

logger = logging.getLogger(__name__)

_MIN_REMAINING_CALLS = 100
_SAVE_EVERY_CALLS = 100

class Dictionary(object):
    """The dictionary of definitions.

    The native representation of the dictionary is a mapping from a word
    to a list of definitions, each of which is a sequence of words. All
    the words are stored as strings.

    """
    def __init__(self, path=None, try_lowercase=False):
        self._data = {}
        self._try_lowercase = try_lowercase
        self._path = path
        self._tmp_path = os.path.join(os.path.dirname(path),
                                         self._path + '.tmp')
        if self._path and os.path.exists(self._path):
            self.load()

    def load(self):
        with open(self._path, 'r') as src:
            self._data = json.load(src)

    def save(self):
        logger.debug("saving...")
        with open(self._tmp_path, 'w') as dst:
            json.dump(self._data, dst, indent=2)
        shutil.move(self._tmp_path, self._path)
        logger.debug("saved")

    def _wait_until_quota_reset(self):
        while True:
            status = self._account_api.getApiTokenStatus()
            logger.debug("{} remaining calls".format(status.remainingCalls))
            if status.remainingCalls >= _MIN_REMAINING_CALLS:
                logger.debug("Wordnik quota was resetted")
                self._remaining_calls = status.remainingCalls
                return
            logger.debug("sleep until quota reset")
            time.sleep(60.)

    def setup_identity_mapping(self, vocab):
        for word in vocab.words:
            self._data[word] = [[word]]
        self.save()

    def setup_spelling(self, vocab):
        for word in vocab.words:
            def_ = word.decode('utf-8')
            if len(def_) > 10:
                def_ = u"{}-{}".format(def_[:5], def_[-5:])
            self._data[word] = [list(def_)]
        self.save()

    def add_from_lemma_definitions(self, vocab):
        """Add lemma definitions for non-lemmas.

        This code covers the following scenario: supposed a dictionary is crawled,
        but only for word lemmas.

        """
        lemmatizer = nltk.WordNetLemmatizer()
        for word in vocab.words:
            try:
                for part_of_speech in ['a', 's', 'r', 'n', 'v']:
                    lemma = lemmatizer.lemmatize(word, part_of_speech)
                    lemma_defs = self._data.get(lemma)
                    if lemma != word and lemma_defs:
                        # This can be quite slow. But this code will not be used
                        # very often.
                        for def_ in lemma_defs:
                            if not def_ in self._data[word]:
                                self._data[word].append(def_)
            except:
                logger.error("lemmatizer crashed on {}".format(word))
        self.save()

    def crawl_lemmas(self, vocab):
        """Add Wordnet lemmas as definitions."""
        lemmatizer = nltk.WordNetLemmatizer()
        for word in vocab.words:
            definitions = []
            try:
                for part_of_speech in ['a', 's', 'r', 'n', 'v']:
                    lemma = lemmatizer.lemmatize(word, part_of_speech)
                    if lemma != word and not [lemma] in definitions:
                        definitions.append([lemma])
            except:
                logger.error("lemmatizer crashed on {}".format(word))
            if definitions:
                self._data[word] = definitions
        self.save()

    def crawl_lowercase(self, vocab):
        """Add Wordnet lemmas as definitions."""
        for word in vocab.words:
            self._data[word] = [[word.lower()]]
        self.save()

    def crawl_wordnik(self, vocab, api_key, call_quota=15000, crawl_also_lowercase=False):
        """Download and preprocess definitions from Wordnik.

        vocab
            Vocabulary for which the definitions should be found.
        api_key
            The API key to use in communications with Wordnik.
        call_quota
            Maximum number of calls per hour.

        """
        self._remaining_calls = call_quota
        self._last_saved = 0

        client = swagger.ApiClient(
            api_key, 'https://api.wordnik.com/v4')
        self._word_api = WordApi.WordApi(client)
        self._account_api = AccountApi.AccountApi(client)
        toktok = nltk.ToktokTokenizer()

        words = list(vocab.words)

        if crawl_also_lowercase:
            logger.info("Adding lowercase words to crawl")
            lowercased = []
            for w in words:
                if w.lower() not in vocab._word_to_id:
                    lowercased.append(w.lower())
            logger.info("Crawling additional {} words".format(len(lowercased)))
            words.extend(sorted(lowercased))

        # Here, for now, we don't do any stemming or lemmatization.
        # Stemming is useless because the dictionary is not indexed with
        # lemmas, not stems. Lemmatizers, on the other hand, can not be
        # fully trusted when it comes to unknown words.
        for word in words:
            if word in self._data:
                logger.debug("a known word {}, skip".format(word))
                continue

            if self._last_saved >= _SAVE_EVERY_CALLS:
                self.save()
                self._last_saved = 0

            # 100 is a safery margin, I don't want to DDoS Wordnik :)
            if self._remaining_calls < _MIN_REMAINING_CALLS:
                self._wait_until_quota_reset()
            try:
                definitions = self._word_api.getDefinitions(word)
            except Exception:
                logger.error("error during fetching '{}'".format(word))
                logger.error(traceback.format_exc())
                continue
            self._remaining_calls -= 1
            self._last_saved += 1

            if not definitions:
                definitions = []
            self._data[word] = []
            for def_ in definitions:
                # TODO(kudkudak): Not super sure if lowering here is correct if we don't lower in
                # normal text
                # Note: should be fine when def has separate lookup and if initialzed from raw embeddinggs
                # uses some fallback strategy
                tokenized_def = [token.encode('utf-8')
                                 for token in toktok.tokenize(def_.text.lower())]
                self._data[word].append(tokenized_def)
            logger.debug("definitions for '{}' fetched {} remaining".format(word, self._remaining_calls))
        self.save()
        self._last_saved = 0

    def num_entries(self):
        return len(self._data)

    def get_definitions(self, key):
        if key not in self._data and self._try_lowercase:
            return self._data.get(key.lower(), [])
        else:
            return self._data.get(key, [])


class Retrieval(object):

    def __init__(self, vocab, dictionary,
                 max_def_length=1000, exclude_top_k=None):
        """Retrieves the definitions.

        vocab
            The vocabulary.
        dictionary
            The dictionary of the definitions.
        max_def_length
            Disregard definitions that are longer than that.
        exclude_top_k
            Do not provide defitions for the first top k
            words of the vocabulary (typically the most frequent ones).
        """
        self._vocab = vocab
        self._dictionary = dictionary
        self._max_def_length = max_def_length
        self._exclude_top_k = exclude_top_k

        # Preprocess all the definitions to see token ids instead of chars

    def retrieve(self, batch):
        """Retrieves all definitions for a batch of words sequences.

        TODO: definitions of phrases, phrasal verbs, etc.

        Returns
        -------
        defs
            A list of word definitions, each definition is a list of words.
        def_map
            A list of triples (batch_index, time_step, def_index). Maps
            words to their respective definitions from `defs`.

        """
        definitions = []
        def_map = []
        word_def_indices = defaultdict(list)

        for seq_pos, sequence in enumerate(batch):
            for word_pos, word in enumerate(sequence):
                if isinstance(word, numpy.ndarray):
                    word = vec2str(word)
                word_id = self._vocab.word_to_id(word)
                if (self._exclude_top_k
                    and word_id != self._vocab.unk
                    and word_id < self._exclude_top_k):
                    continue

                if word not in word_def_indices:
                    # The first time a word is encountered in a batch
                    word_defs = self._dictionary.get_definitions(word)
                    if not word_defs:
                        # No defition for this word
                        continue
                    for i, def_ in enumerate(word_defs):
                        if len(def_) > self._max_def_length:
                            continue
                        final_def_ = [self._vocab.bod]
                        for token in def_:
                            final_def_.append(self._vocab.word_to_id(token))
                        final_def_.append(self._vocab.eod)
                        word_def_indices[word].append(len(definitions))
                        definitions.append(final_def_)
                for def_index in word_def_indices[word]:
                    def_map.append((seq_pos, word_pos, def_index))

        return definitions, def_map

    def retrieve_and_pad(self, batch):
        defs, def_map = self.retrieve(batch)
        if not defs:
            defs.append(self.sentinel_definition())
        # `defs` have variable length and have to be padded
        max_def_length = max(map(len, defs))
        def_array = numpy.zeros((len(defs), max_def_length), dtype='int64')
        def_mask = numpy.ones_like(def_array, dtype='float32')
        for i, def_ in enumerate(defs):
            def_array[i, :len(def_)] = def_
            def_mask[i, len(def_):] = 0.
        def_map = (numpy.array(def_map)
                   if def_map
                   else numpy.zeros((0, 3), dtype='int64'))
        return def_array, def_mask, def_map

    def sentinel_definition(self):
        """An empty definition.

        If you ever need a definition that is syntactically correct but
        doesn't mean a thing, call me.

        """
        return [self._vocab.bod, self._vocab.eod]
