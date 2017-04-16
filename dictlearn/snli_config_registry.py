from dictlearn.config_registry import ConfigRegistry

snli_config_registry = ConfigRegistry()
snli_config_registry.set_root_config({
    'data_path':  '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/train.h5',
    'layout': 'snli',
    'embedding_path': '/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.h5',

    'translate_dim': 300,
    'emb_dim': 300,
    "encoder": "sum",
    "dropout": 0.2,
    'batch_size': 32,
    'lr': 0.1,

    # Misc
    'monitor_parameters': 0,
    'mon_freq_train': 10000,
    'save_freq_batches': 10000,
    'mon_freq_valid': 10000,
    'batch_size_valid': 32,
    'n_batches': 100000
})