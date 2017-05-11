from dictlearn.config_registry import ConfigRegistry

qa_config_registry = ConfigRegistry()
qa_config_registry.set_root_config({
    # data
    'data_path' : "",
    'dict_path' : "",
    'vocab_path' : "",
    'dict_vocab_path' : "",
    'embedding_path' : "",
    'layout' : 'standard',
    'num_input_words' : 10000,
    'def_num_input_words' : 0,
    'max_length' : 100,
    'batch_size' : 32,
    'batch_size_valid' : 32,
    'max_def_length' : 1000,
    'exclude_top_k' : 0,

    # model
    'def_reader' : 'LSTMReadDefinitions',
    'dim' : 128,
    'emb_dim' : 0,
    'readout_dims' : [],
    'coattention' : True,
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'grad_clip_threshold' : 5.0,
    'dropout' : 0.,
    'random_unk' : False,
    'def_word_gating' : "none",
    'compose_type' : "sum",
    'reuse_word_embeddings' : False,
    'train_only_def_part' : False,

    # monitoring and checkpointing
    'mon_freq_train' : 10,
    'save_freq_batches' : 1000,
    'save_freq_epochs' : 1,
    'n_batches' : 0,
    'monitor_parameters' : False
})

c = qa_config_registry['root']
c['data_path'] = 'squad/squad_from_scratch'
c['layout'] = 'squad'
c['mon_freq_train'] = 100
c['grad_clip_threshold'] = 50.
qa_config_registry['squad'] = c

c = qa_config_registry['squad']
c['data_path'] = 'squad/squad_glove'
c['emb_dim'] = 300
c['embedding_path'] = 'squad/squad_glove/glove_w_specials.npy'
c['num_input_words'] = 0
qa_config_registry['squad_glove'] = c

def from1to2(c):
    c['batch_size'] = 128
    c['batch_size_valid'] = 128
    c['dim'] = 200
    return c

c = qa_config_registry['squad']
from1to2(c)
qa_config_registry['squad2'] = c

c = qa_config_registry['squad_glove']
from1to2(c)
qa_config_registry['squad_glove2'] = c

def from2to3(c):
    c['max_def_length'] = 30
    c['exclude_top_k'] = 10000
    c['dict_path'] = 'squad/squad_from_scratch/dict.json'
    return c

c = qa_config_registry['squad2']
from2to3(c)
qa_config_registry['squad3'] = c

c = qa_config_registry['squad_glove2']
from2to3(c)
c['num_input_words'] = 0
qa_config_registry['squad_glove3'] = c

def from3to4(c):
    c['num_input_words'] = 3000
    c['exclude_top_k'] = 3000
    c['emb_dim'] = 300
    c['reuse_word_embeddings'] = True
    c['compose_type'] = 'transform_and_sum'
    c['dict_path'] = 'squad/squad_from_scratch/dict2.json'
    return c

c = qa_config_registry['squad3']
from3to4(c)
qa_config_registry['squad4'] = c

c = qa_config_registry['squad_glove3']
from3to4(c)
c['num_input_words'] = 0
qa_config_registry['squad_glove4'] = c

def from4to5(c):
    c['dict_path'] = 'squad/squad_from_scratch/dict_wordnet3.2.json'
    c['batch_size'] = 32
    return c

qa_config_registry['squad5'] = from4to5(qa_config_registry['squad4'])
qa_config_registry['squad_glove5'] = from4to5(qa_config_registry['squad_glove4'])

def tune_depth_and_dropout(c):
    c['readout_dims'] = [200]
    c['dropout'] = 0.2
    return c

qa_config_registry['squad6'] = tune_depth_and_dropout(qa_config_registry['squad5'])
qa_config_registry['squad_glove6'] = tune_depth_and_dropout(qa_config_registry['squad_glove2'])
qa_config_registry['squad_glove7'] = tune_depth_and_dropout(qa_config_registry['squad_glove5'])
