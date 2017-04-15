import tensorflow

from blocks.extensions import SimpleExtension
from blocks.serialization import load, load_parameters
from blocks.extensions.saveload import Load

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
