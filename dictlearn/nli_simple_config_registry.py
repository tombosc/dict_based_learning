from dictlearn.config_registry import ConfigRegistry

snli_config_registry = ConfigRegistry()

# Each epoch has ~500k examples
snli_config_registry.set_root_config({
    'data_path':  '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/',
    'layout': 'snli',
    'try_lowercase': True,

    # Lookup params
    'translate_dim': 300,
    'max_def_per_word': 100000,
    'mlp_dim': 600,
    'emb_dim': 300,
    'dict_path': '',
    # Remove by default embeddings. Our goal ATM is to beat random init
    'embedding_path': '', #/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.npy',
    'compose_type': '',
    'disregard_word_embeddings': False,
    'exclude_top_k': -1,
    'max_def_length': 50,
    'train_emb': 1, # Remove by default embeddings. Our goal ATM is to beat random init
    "combiner_dropout": 0.0,
    "combiner_dropout_type": "per_unit",
    "combiner_gating": "none",
    "combiner_shortcut": False,
    'reader_type': 'rnn',
    'share_def_lookup': False,
    'combiner_bn': False,

    'num_input_words': 0, # Will take vocab size
    "encoder": "sum",
    "dropout": 0.3,
    'batch_size': 512,
    'lr': 0.001,
    'l2': 4e-6,

    # Misc
    'monitor_parameters': 0,
    'mon_freq_train': 1000,
    'save_freq_batches': 1000,
    'mon_freq_valid': 1000,
    'n_batches': 200000 # ~200 epochs of SNLI
})

### Establish baseline ###
c = snli_config_registry['root']
snli_config_registry['baseline'] = c
c = snli_config_registry['root']
c['data_path'] = '/data/lisa/exp/jastrzes/dict_based_learning/data/mnli/'
snli_config_registry['baseline_mnli'] = c

### Small dict ###
# Looking up words from test/dev as well
c['dict_path'] = '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/dict_all.json'
c['exclude_top_k'] = 2500
c['share_def_lookup'] = False
c['reader_type'] = 'mean'
c['combiner_dropout'] = 0.5
c['combiner_dropout_type'] = "per_unit" # Safer choice
c['train_emb'] = 1
c['embedding_path'] = ''
c['lr'] = 0.0006 # Note: 0.001 works better
c['num_input_words'] = 5000
c['compose_type'] = 'sum' # Forces to use dict
c['disregard_word_embeddings'] = False
snli_config_registry['sum_small_dict'] = c

### RNN + Small dict ###
c = snli_config_registry['sum_small_dict']
c['reader_type'] = 'rnn'
snli_config_registry['rnn_small_dict'] = c
