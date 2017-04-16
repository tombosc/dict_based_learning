from dictlearn.config_registry import ConfigRegistry

snli_config_registry = ConfigRegistry()
snli_config_registry.set_root_config({
    'data_path':  '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/',
    'layout': 'snli',
    'embedding_path': '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.npy',

    'translate_dim': 300,
    'emb_dim': 300,
    "encoder": "sum",
    "dropout": 0.2,
    'batch_size': 512,
    'lr': 0.1,
    'l2': 4e-6,

    # Misc
    'monitor_parameters': 0,
    'mon_freq_train': 100,
    'save_freq_batches': 1000,
    'mon_freq_dev': 1000,
    'batch_size_dev': 512,
    'n_batches': 100000
})