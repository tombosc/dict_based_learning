from dictlearn.config_registry import ConfigRegistry

prefix = '/data/lisa/exp/bosctom/dict_based_learning/'

lm_config_registry = ConfigRegistry()
lm_config_registry.set_root_config({
    # data
    'data_path': prefix + 'data/onebillionword/',
    'layout': 'standard',
    'dict_path' : "",
    'layout' : 'standard',
    'num_input_words' : 10000,
    'num_output_words': 10000,
    'max_length' : 100,
    'batch_size' : 64,
    'batch_size_valid' : 64,
    'max_def_length' : 100,
    'exclude_top_k' : -1,
    'top_k_words' : -1,

    # model
    'dim' : 500,
    'compose_type' : 'sum',
    'standalone_def_rnn' : False,
    'standalone_def_lookup': False,
    'disregard_word_embeddings' : False,
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'grad_clip_threshold' : 5.0,

    # monitoring and checkpointing
    'mon_freq_train' : 100,
    'mon_freq_valid' : 1000,
    'save_freq_batches' : 1000,
    'n_batches' : 0,
    'monitor_parameters' : False
})



# Without dictionary
c = lm_config_registry['root']
c['num_output_words'] = 30000
c['learning_rate'] = 0.0003
c['batch_size'] = 32
c['batch_size_valid'] = 32
lm_config_registry['obw_base_30k_slower'] = c

c = lm_config_registry['root']
c['num_output_words'] = 10000
c['batch_size'] = 32
c['batch_size_valid'] = 32
c['learning_rate'] = 0.0003
lm_config_registry['obw_base_10k_slower'] = c

# With Stan's SNLI dict, doing mean combiner
c = lm_config_registry['root']
c['num_input_words'] = 10000
c['num_output_words'] = 10000
c['exclude_top_k'] = 10000
c['batch_size'] = 32
c['batch_size_valid'] = 32
c['max_def_per_word'] = 3
c['max_def_length'] = 100
c['dict_path'] = prefix + 'data/dict_all.json'
c['learning_rate'] = 0.0003 
# Stuff to tune:
c['standalone_def_lookup'] = False
c['standalone_def_rnn'] = False
c['compose_type'] = 'linear_and_sum'
lm_config_registry['obw_10k_dict1'] = c

c['standalone_def_rnn'] = True
c['compose_type'] = 'linear_and_sum'
lm_config_registry['obw_10k_dict2'] = c

c['standalone_def_rnn'] = False
c['compose_type'] = 'sum'
lm_config_registry['obw_10k_dict3'] = c

c['standalone_def_rnn'] = True
c['compose_type'] = 'sum'
lm_config_registry['obw_10k_dict4'] = c

