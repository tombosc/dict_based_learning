from dictlearn.config_registry import ConfigRegistry

snli_config_registry = ConfigRegistry()

# Each epoch has ~500k examples
snli_config_registry.set_root_config({
    'data_path':  '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/',
    'layout': 'snli',

    # Lookup params
    'translate_dim': 300,
    'emb_dim': 300,
    'dict_path': '',
    'embedding_path': '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.npy',
    'compose_type': '',
    'disregard_word_embeddings': False,
    'exclude_top_k': -1,
    'max_def_length': 1000,
    'train_emb': 0,

    'num_input_words': 0, # Will take vocab size
    "encoder": "sum",
    "dropout": 0.2,
    'batch_size': 512,
    'lr': 0.02,
    'l2': 4e-6,

    # Misc
    'monitor_parameters': 0,
    'mon_freq_train': 1000,
    'save_freq_batches': 1000,
    'mon_freq_dev': 1000,
    'batch_size_dev': 512,
    'n_batches': 100000 # ~100 epochs of SNLI
})

c = snli_config_registry['root']
c['dict_path'] = '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/dict.json'
c['exclude_top_k'] = 10000
c['compose_type'] = 'fully_connected_linear' # Affine transformation
c['disregard_word_embeddings'] = False
snli_config_registry['small_dict'] = c