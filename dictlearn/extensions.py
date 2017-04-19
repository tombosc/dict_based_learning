import os
import signal
import time
import subprocess
import atexit
import logging
import cPickle

import tensorflow

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

    def __init__(self, stream, stream_path, *args, **kwargs):
        self._stream = stream
        self._stream_path = stream_path
        super(StartFuelServer, self).__init__(*args, **kwargs)

    def do(self, *args, **kwars):
        with open(self._stream_path, 'w') as dst:
            cPickle.dump(self._stream, dst, 0)
        port = get_free_port()
        self.main_loop.data_stream.port = port
        logger.debug("Starting the Fuel server...")
        ret = subprocess.Popen(
            ["start_fuel_server.py", self._stream_path, str(port), '100'])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))
