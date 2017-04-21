import os
import signal
import time
import subprocess
import atexit
import logging
import cPickle
import tensorflow
import pandas as pd
import scipy
import numpy as np

from collections import defaultdict
from six import iteritems

from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.serialization import load, load_parameters
from blocks.extensions.saveload import Load

from dictlearn.util import get_free_port

logger = logging.getLogger(__name__)

def construct_dict_embedder(theano_fnc, vocab, retrieval):
    """
    Parameters
    ----------
    theano_fnc: theano.Function
        (batch_size, seq_len) -> (batch_size, seq_len, word_dim)

    vocab: Vocabulary
        Vocabulary instance

    Returns
    -------
        Python function: (batch_size, ) -> (batch_size, word_dim)
    """

    def _embedder(word_list):
        word_ids = vocab.encode(word_list)
        word_ids = np.array(word_ids)
        word_ids = word_ids.reshape((-1, 1)) # Just to adhere to theano.Function, whatever
        def_array, def_mask, def_map = retrieval.retrieve_and_pad(np.array(word_list).reshape(-1,))
        import pdb;
        pdb.set_trace()
        word_vectors = theano_fnc(word_ids, def_array, def_mask, def_map)
        word_vectors = word_vectors.reshape((len(word_list), -1))
        import pdb;
        pdb.set_trace()
        return word_vectors

    return _embedder

def construct_embedder(theano_fnc, vocab):
    """
    Parameters
    ----------
    theano_fnc: theano.Function
        (batch_size, seq_len) -> (batch_size, seq_len, word_dim)

    vocab: Vocabulary
        Vocabulary instance

    Returns
    -------
        Python function: (batch_size, ) -> (batch_size, word_dim)
    """

    def _embedder(word_list):
        word_ids = vocab.encode(word_list)
        word_ids = np.array(word_ids)
        word_ids = word_ids.reshape((-1, 1)) # Just to adhere to theano.Function, whatever
        word_vectors = theano_fnc(word_ids)
        word_vectors = word_vectors.reshape((len(word_list), -1))
        return word_vectors

    return _embedder


def evaluate_similarity(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """

    from web.embedding import Embedding

    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])
    B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])
    scores = np.array([v1.dot(v2.T) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


class SimilarityWordEmbeddingEval(SimpleExtension):
    """
    Parameters
    ----------

    embedder: function: word -> vector
    """

    def __init__(self, embedder, prefix="", **kwargs):
        try:
            from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
        except ImportError:
            raise RuntimeError("Please install web (https://github.com/kudkudak/word-embeddings-benchmarks)")

        self._embedder = embedder
        self._prefix = prefix

        # Define tasks
        logger.info("Downloading benchmark data")
        tasks = { # TODO: Pick a bit better tasks
            "MEN": fetch_MEN(),
            "WS353": fetch_WS353(),
            "SIMLEX999": fetch_SimLex999()
        }

        # Print sample data
        for name, data in iteritems(tasks):
            logger.info(
            "Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1],
                data.y[0]))

        logger.info("Checking embedder for " + prefix)
        logger.info(embedder(["love"])) # Test embedder

        self._tasks = tasks

        super(SimilarityWordEmbeddingEval, self).__init__(**kwargs)

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if '_embedder' in dict_:
            del dict_['_embedder']
        if '_tasks' in dict_:
            del dict_['_tasks']
        return dict_

    def add_records(self, log, record_tuples):
        """Helper function to add monitoring records to the log."""
        for name, value in record_tuples:
            if not name:
                raise ValueError("monitor variable without name")
            log.current_row[name] = value

    def do(self, *args, **kwargs):
        # Embedd
        all_words = []
        for task in self._tasks:
            for row in self._tasks[task].X:
                for w in row:
                    all_words.append(w)
        logger.info("Embedding vectors")
        all_words_vectors = self._embedder(all_words)
        W = dict(zip(np.array(all_words).reshape((-1,)), all_words_vectors))

        # Calculate results using helper function
        record_items = []
        for name, data in iteritems(self._tasks):
            eval = evaluate_similarity(W, data.X, data.y)
            record_items.append((self._prefix + "_" + name, eval))

        self.add_records(self.main_loop.log, record_items)


class DumpTensorflowSummaries(SimpleExtension):
    def __init__(self, save_path, **kwargs):
        self._save_path = save_path
        super(DumpTensorflowSummaries, self).__init__(**kwargs)

    @property
    def file_writer(self):
        if not hasattr(self, '_file_writer'):
            self._file_writer = tensorflow.summary.FileWriter(
                self._save_path, flush_secs=10.)
        return self._file_writer

    def __getstate__(self):
        # FileWriter from TensorFlow is not picklable
        dict_ = self.__dict__
        if '_file_writer' in dict_:
            del dict_['_file_writer']
        return dict_

    def do(self, *args, **kwargs):
        summary = tensorflow.Summary()
        for key, value in self.main_loop.log.current_row.items():
            try:
                float_value = float(value)
                value = summary.value.add()
                value.tag = key
                value.simple_value = float_value
            except:
                pass
        self.file_writer.add_summary(
            summary, self.main_loop.log.status['iterations_done'])


class DumpCSVSummaries(SimpleExtension):
    def __init__(self, save_path, mode="w", **kwargs):
        self._save_path = save_path
        self._mode = mode

        if self._mode == "w":
            # Clean up file
            with open(os.path.join(self._save_path, "logs.csv"), "w") as f:
                pass
            self._current_log = defaultdict(list)
        else:
            self._current_log = pd.read_csv(os.path.join(self._save_path, "logs.csv")).to_dict()

        super(DumpCSVSummaries, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        for key, value in self.main_loop.log.current_row.items():
            try:
                float_value = float(value)

                if not key.startswith("val") and not key.startswith("train") and not key.startswith("test"):
                    key = "train_" + key

                self._current_log[key].append(float_value)
            except:
                pass

        # Make sure all logs have same length (for csv serialization)
        max_len = max([len(v) for v in self._current_log.values()])
        for k in self._current_log:
            if len(self._current_log[k]) != max_len:
                self._current_log[k] += [self._current_log[k][-1] for _ in range(max_len - len(self._current_log[k]))]

        pd.DataFrame(self._current_log).to_csv(os.path.join(self._save_path, "logs.csv"))



class LoadNoUnpickling(Load):
    """Like `Load` but without unpickling.

    Avoids unpiclkling the main loop by assuming that the log
    and the iteration state were saved separately.

    """

    def load_to(self, main_loop):
        with open(self.path, "rb") as source:
            main_loop.model.set_parameter_values(load_parameters(source))
            if self.load_iteration_state:
                main_loop.iteration_state = load(source, name='iteration_state')
            if self.load_log:
                main_loop.log = load(source, name='log')


class StartFuelServer(SimpleExtension):

    def __init__(self, stream, stream_path, script_path="start_fuel_server.py", hwm=100, *args, **kwargs):
        self._stream = stream
        self._hwm = hwm
        self._stream_path = stream_path
        self._script_path = script_path
        super(StartFuelServer, self).__init__(*args, **kwargs)

    def do(self, *args, **kwars):
        with open(self._stream_path, 'w') as dst:
            cPickle.dump(self._stream, dst, 0)
        port = get_free_port()
        self.main_loop.data_stream.port = port
        logger.debug("Starting the Fuel server on port " + str(port))
        ret = subprocess.Popen(
            [self._script_path,
                self._stream_path, str(port), str(self._hwm)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))
