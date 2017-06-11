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

    # retrieval hacks
    'max_def_length' : 1000,
    'with_too_long_defs' : 'drop',
    'max_def_per_word' : 1000,
    'with_too_many_defs' : 'random',
    'exclude_top_k' : 0,

    # model
    'def_reader' : 'LSTMReadDefinitions',
    'dim' : 128,
    'emb_dim' : 0,
    'readout_dims' : [],
    'coattention' : True,
    'init_width' : 0.1,
    'rec_init_width' : 0.1,
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'annealing_learning_rate' : 0.0001,
    'annealing_start_epoch' : 10,
    'grad_clip_threshold' : 5.0,
    'emb_dropout' : 0.0,
    'emb_dropout_type' : 'regular',
    'dropout' : 0.,
    'dropout_type' : 'regular',
    'random_unk' : False,
    'def_word_gating' : "none",
    'compose_type' : "sum",
    'reuse_word_embeddings' : False,
    'train_only_def_part' : False,

    # monitoring and checkpointing
    'mon_freq_train' : 10,
    'save_freq_batches' : 1000,
    'save_freq_epochs' : 1,
    # that corresponds to about 12 epochs
    'n_batches' : 0,
    'n_epochs' : 12,
    'monitor_parameters' : False
})
qar = qa_config_registry

c = qar['root']
c['data_path'] = 'squad/squad_from_scratch'
c['layout'] = 'squad'
c['mon_freq_train'] = 100
c['grad_clip_threshold'] = 50.
qar['squad'] = c

c = qar['squad']
c['data_path'] = 'squad/squad_glove'
c['emb_dim'] = 300
c['embedding_path'] = 'squad/squad_glove/glove_w_specials.npy'
c['num_input_words'] = 0
qar['squad_glove'] = c

def from1to2(c):
    c['batch_size'] = 128
    c['batch_size_valid'] = 128
    c['dim'] = 200
    return c

c = qar['squad']
from1to2(c)
qar['squad2'] = c

c = qar['squad_glove']
from1to2(c)
qar['squad_glove2'] = c

def from2to3(c):
    c['max_def_length'] = 30
    c['exclude_top_k'] = 10000
    c['dict_path'] = 'squad/squad_from_scratch/dict.json'
    return c

c = qar['squad2']
from2to3(c)
qar['squad3'] = c

c = qar['squad_glove2']
from2to3(c)
c['num_input_words'] = 0
qar['squad_glove3'] = c

def from3to4(c):
    c['num_input_words'] = 3000
    c['exclude_top_k'] = 3000
    c['emb_dim'] = 300
    c['reuse_word_embeddings'] = True
    c['compose_type'] = 'transform_and_sum'
    c['dict_path'] = 'squad/squad_from_scratch/dict2.json'
    return c

c = qar['squad3']
from3to4(c)
qar['squad4'] = c

c = qar['squad_glove3']
from3to4(c)
c['num_input_words'] = 0
qar['squad_glove4'] = c

def from4to5(c):
    c['dict_path'] = 'squad/squad_from_scratch/dict_wordnet3.2.json'
    c['batch_size'] = 32
    return c

qar['squad5'] = from4to5(qar['squad4'])
qar['squad_glove5'] = from4to5(qar['squad_glove4'])

def tune_depth_and_dropout(c):
    c['readout_dims'] = [200]
    c['dropout'] = 0.2
    return c

# a better regularized baseline config
c = qar['squad2']
c['batch_size'] = 32
qar['squad7'] = c

# spelling
def change_dict_to_spelling(c):
    c['dict_path'] = 'squad/squad_from_scratch/dict_spelling2.json'
    # BUG: should be vocab_chars.txt instead
    c['dict_vocab_path'] = 'squad/squad_from_scratch/vocab_chars.txt'
    return c
c = change_dict_to_spelling(qar['squad5'])
# for some reason this was helpful
# TODO: try again without it
c['emb_dim'] = 200
qar['squad8'] = c


# dict + spelling
c = qar['squad5']
c['dict_path'] = 'squad/squad_from_scratch/dict_wordnet4.3.json'
c['vocab_path'] = 'squad/squad_from_scratch/vocab_with_chars.txt'
c['num_input_words'] = 3100
c['exclude_top_k'] = 3100
qar['squad9'] = c

# PAPER CONFIGS

# the baseline without embeddings
qar['squad10'] = tune_depth_and_dropout(qar['squad7'])
# the baseline with spelling
qar['squad11'] = tune_depth_and_dropout(qar['squad8'])
c = qar['squad11']
# fix the old hack with the dimensionality
c['emb_dim'] = 300
qar['squad11b'] = c
# the baseline with the dictionary only
qar['squad6'] = tune_depth_and_dropout(qar['squad5'])
# dictionary + spelling
qar['squad13'] = tune_depth_and_dropout(qar['squad9'])

# the glove baseline
c = tune_depth_and_dropout(qar['squad_glove2'])
c['batch_size'] = 32
qar['squad_glove6'] = c

# glove + dictionary
c = tune_depth_and_dropout(qar['squad_glove5'])
c['compose_type'] = 'gated_transform_and_sum'
qar['squad_glove7'] = c

# can we use separate embeddings for def reader?
c = qar['squad_glove7']
c['reuse_word_embeddings'] = False
c['def_num_input_words'] = 3000
c['dict_vocab_path'] = 'squad/squad_from_scratch/vocab_dict_wordnet3.2.txt'
qar['squad_glove8'] = c

# a baseline: what if we use just lemmas as definitions?
c = qar['squad_glove7']
c['dict_path'] = 'squad/squad_from_scratch/dict_lemmalower.json'
qar['squad_glove9'] = c
# possible todo:
# glove + dict + spelling
# recursion
# bigrams

# POST PAPER CONFIGS

# try to change the set of words that we exclude
c = qar['squad13']
c['exclude_top_k'] = 0
c['max_def_per_word'] = 30
c['with_too_many_defs'] = 'exclude'
qar['squad14'] = c

# dropout of embeddings
c = qar['squad10']
c['emb_dropout_type'] = 'same_mask'
c['emb_dropout'] = 0.5
c['n_epochs'] = 30
c['annealing_start_epoch'] = 20
qar['squad15'] = c

# hyperparameter cleanup
c = qar['squad15']
c['dropout_type'] = 'same_mask'
c['init_width'] = 0.
c['rec_init_width'] = 0.
qar['squad16'] = c

c = qar['squad_glove6']
c['init_width'] = 0.
c['rec_init_width'] = 0.
c['dropout_type'] = 'same_mask'
c['emb_dropout_type'] = 'same_mask'
c['emb_dropout'] = 0.5
c['n_epochs'] = 30
c['annealing_start_epoch'] = 20
qar['squad_glove10'] = c
