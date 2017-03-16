import h5py

from fuel.datasets.hdf5 import H5PYDataset

def text_to_h5py_dataset(text_path, dst_path):
    # The simplest is to load everything to memory first.
    # If memory becomes an issue, this code can be optimized.
    words = []
    with open(text_path, 'r') as src:
        for line in src:
            words.extend(line.strip().split())

    with h5py.File(dst_path, 'w') as dst:
        dtype = h5py.special_dtype(vlen=bytes)
        table = dst.create_dataset('words', (len(words),), dtype=dtype)
        table[:] = words

        dst.attrs['split'] = H5PYDataset.create_split_array({
                'train' : {
                    'words' : (0, len(words))
                }
            })
