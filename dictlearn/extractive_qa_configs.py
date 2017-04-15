from dictlearn.config_registry import ConfigRegistry

qa_config_registry = ConfigRegistry()
qa_config_registry.set_root_config({
    # data
    'data_path' : "",
    'dict_path' : "",
    'embedding_path' : "",
    'layout' : 'standard',
    'num_input_words' : 10000,
    'max_length' : 100,
    'batch_size' : 32,
    'batch_size_valid' : 32,

    # model
    'dim' : 128,
    'emb_dim' : 0,
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'grad_clip_threshold' : 5.0,

    # monitoring and checkpointing
    'mon_freq_train' : 10,
    'mon_freq_valid' : 1000,
    'save_freq_batches' : 1000,
    'n_batches' : 0,
    'monitor_parameters' : False
})

c = qa_config_registry['root']
c['data_path'] = '/data/lisatmp4/bahdanau/data/squad/squad_from_scratch'
c['layout'] = 'squad'
c['mon_freq_train'] = 100
c['grad_clip_threshold'] = 50.
qa_config_registry['squad'] = c

c = qa_config_registry['squad']
c['data_path'] = '/data/lisatmp4/bahdanau/data/squad/squad_glove'
c['emb_dim'] = 300
c['embedding_path'] = '/data/lisatmp4/bahdanau/data/squad/squad_glove/glove_w_specials.npy'
c['num_input_words'] = 0
qa_config_registry['squad_glove'] = c

def from1to2(c):
    c['batch_size'] = 256
    c['batch_size_valid'] = 256
    c['dim'] = 200

c = qa_config_registry['squad']
from1to2(c)
qa_config_registry['squad2'] = c

c = qa_config_registry['squad_glove']
from1to2(c)
qa_config_registry['squad_glove2'] = c
