import os
import signal
import time
import subprocess
import atexit
import logging
import cPickle
import tensorflow
import pandas as pd

from collections import defaultdict

from blocks.extensions import SimpleExtension
from blocks.serialization import load, load_parameters
from blocks.extensions.saveload import Load

from dictlearn.util import get_free_port

logger = logging.getLogger(__name__)


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
        port = 5557 #get_free_port()
        self.main_loop.data_stream.port = port
        logger.debug("Starting the Fuel server on port " + str(port))
        ret = subprocess.Popen(
            [self._script_path,
                self._stream_path, str(port), str(self._hwm)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))
