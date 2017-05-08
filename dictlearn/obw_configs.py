from dictlearn.config_registry import ConfigRegistry

lm_config_registry = ConfigRegistry()
lm_config_registry.set_root_config({
    # data
    'data_path': 'onebillionword/',
    'dict_path' : "",
    'layout' : 'standard',
    'num_input_words' : 10000,
    'num_output_words': 10000,
    'max_length' : 100,
    'batch_size' : 64,
    'batch_size_valid' : 64,
    'max_def_length' : 100,
    'max_def_per_word': 1000,
    'exclude_top_k' : -1,
    'top_k_words' : -1,

    # model
    'emb_dim' : 500,
    'dim' : 500,
    'compose_type' : 'sum',
    'disregard_word_embeddings' : False,
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'grad_clip_threshold' : 5.0,
    # model: def_reader
    'def_reader': 'LSTM',
    'standalone_def_rnn' : False,
    'standalone_def_lookup': False,

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
c['dict_path'] = 'data/dict_snli.json'
c['learning_rate'] = 0.0003 
# Stuff to tune:
c['standalone_def_lookup'] = False
c['standalone_def_rnn'] = False
c['compose_type'] = 'transform_and_sum'
lm_config_registry['obw_10k_dict1'] = c

c['standalone_def_rnn'] = True
c['compose_type'] = 'transform_and_sum'
lm_config_registry['obw_10k_dict2'] = c

c['standalone_def_rnn'] = False
c['compose_type'] = 'sum'
lm_config_registry['obw_10k_dict3'] = c

c['standalone_def_rnn'] = True
c['compose_type'] = 'sum'
lm_config_registry['obw_10k_dict4'] = c

c = lm_config_registry['obw_base_10k_slower']
c['num_input_words'] = 30000
lm_config_registry['obw_base_10k_slower_30kI'] = c

# Want to try a new dictionary
# doesn't contain everything cause it crashed as well
c = lm_config_registry['obw_10k_dict3']
c['dict_path'] = 'dict_obw.json'
lm_config_registry['obw_10k_dict3_n'] = c

# what happens with mean reader
c = lm_config_registry['obw_10k_dict3']
c['def_reader'] = 'mean'
c['dict_path'] = 'dict_obw.json'
lm_config_registry['obw_10k_dict_mean'] = c


c = lm_config_registry['obw_10k_dict1']
c['learning_rate'] = 0.002
lm_config_registry['obw_10k_dict1_fast'] = c

c = lm_config_registry['obw_base_10k_slower']
c['learning_rate'] = 0.002
lm_config_registry['obw_base_10k_fast'] = c

c = lm_config_registry['obw_10k_dict1_fast']
c['standalone_def_rnn'] = True
lm_config_registry['obw_10k_dict2_fast'] = c

# test : sort by cov then len
c = lm_config_registry['obw_10k_dict_mean']
c['dict_path'] = 'dict_obw_reordered_max1_cov.json'
c['max_def_per_word'] = 1 
lm_config_registry['obw_10k_dict_mean_cov1'] = c

# test: sort by len then cov
c['dict_path'] = 'dict_obw_reordered_max1_len.json'
lm_config_registry['obw_10k_dict_mean_len1'] = c
