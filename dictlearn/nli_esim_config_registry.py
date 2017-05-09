from dictlearn.config_registry import ConfigRegistry

nli_esim_config_registry = ConfigRegistry()

# Each epoch has ~500k examples
# Params copied from https://github.com/NYU-MLL/multiNLI/blob/master/python/util/parameters.py
nli_esim_config_registry.set_root_config({
    'data_path':  '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/',
    'layout': 'snli',

    # Lookup params
    'max_def_per_word': 100000,
    'emb_dim': 300,
    'dim': 300,
    'dict_path': '',

    # Also used in NYU-MLI
    'embedding_path': '',
    'train_emb': 1,

    # Dict params
    'compose_type': '',
    'try_lowercase': True,
    'disregard_word_embeddings': False,
    'exclude_top_k': -1,
    'max_def_length': 50,
    'def_dim': 100,
    'def_emb_dim': -1,
    "combiner_dropout": 0.0,
    "combiner_dropout_type": "per_unit",
    "combiner_gating": "none",
    "combiner_shortcut": False,
    'reader_type': 'rnn',
    'share_def_lookup': False,
    'combiner_bn': False,

    'num_input_words': -1, # Will take vocab size
    "dropout": 0.5,
    'batch_size': 32,
    'lr': 0.0004,

    # Misc. Monitor every 100% of epoch
    'monitor_parameters': 0,
    'mon_freq_train': int((500000) / 32),
    'save_freq_batches':int((500000) / 32),
    'mon_freq_valid': int((500000) / 32),
    'n_batches': 200 * (500000 / 32) # ~200 epochs of SNLI
})

### Establish baseline ###
c = nli_esim_config_registry['root']
nli_esim_config_registry['baseline'] = c

c = nli_esim_config_registry['root']
c['num_input_words'] = 3000
nli_esim_config_registry['baseline_3k'] = c

### Establish baseline with Glove ###
c = nli_esim_config_registry['root']
c['embeddng_path'] = '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.npy'
c['train_emb'] = 0
nli_esim_config_registry['baseline_glove'] = c