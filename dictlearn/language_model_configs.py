from dictlearn.config_registry import ConfigRegistry

lm_config_registry = ConfigRegistry()
lm_config_registry.set_root_config({
    # data
    'data_path' : "",
    'dict_path' : "",
    'layout' : 'standard',
    'top_k_words' : 10000,
    'max_length' : 100,
    'batch_size' : 32,
    'batch_size_valid' : 32,

    # model
    'dim' : 128,
    'standalone_def_rnn' : True,
    'learning_rate' : 0.001,
    'grad_clip_threshold' : 5.0,

    # monitoring and checkpointing
    'mon_freq_train' : 10,
    'mon_freq_valid' : 1000,
    'save_freq_batches' : 1000,
})

c = lm_config_registry['root']
c['data_path'] = '/data/lisatmp4/bahdanau/data/lambada'
c['layout'] = 'lambada'
# In LAMBADA validation and test sets have quite long sentences.
# And this implementation for now is very memory inefficient.
c['batch_size_valid'] = 16
c['top_k_words'] = 60000
c['dim'] = 512
c['mon_freq_train'] = 100
lm_config_registry['lambada'] = c


