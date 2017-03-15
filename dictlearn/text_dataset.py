"""Data.

Fuel provides two solutions for reading text, and both don't seem
suitable:

- TextFile only reads sequentially
- H5PyDataset is built with an assumption that there exists
  a non-overlapping set of examples.

That said, we need a different basic solution.

"""

from fuel.datasets import Dataset


class TextDataset(Dataset):
    """Provides basic access to lines of a text file."""
    provides_sources = ('lines',)
    example_iteration_scheme = None

    def __init__(self, path, **kwargs):
        self._path = path
        super(TextDataset, self).__init__(**kwargs)

    def open(self):
        return open(self._path, 'r')

    def get_data(self, state, request=None):
        return (state.readline().strip().split(),)
