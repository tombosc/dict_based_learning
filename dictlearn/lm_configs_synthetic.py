from dictlearn.config_registry import ConfigRegistry
from dictlearn.language_model_configs import lm_config_registry

prefix = '/data/lisa/exp/bosctom/dict_based_learning/'
#prefix = '/media/Docs/projets/dict_based_learning/'


# With dictionary
c = lm_config_registry['root']
c['standalone_def_lookup'] = True
c['standalone_def_rnn'] = True
c['data_path'] = prefix + 'data/toy_data_1'
c['dict_path'] = prefix + 'data/toy_data_1/dict.json'
c['batch_size'] = 16
c['batch_size_valid'] = 16
c['top_k_words'] = 100
c['dim'] = 100 # Features_size * Markov_order
c['mon_freq_train'] = 100
c['mon_freq_valid'] = 1000
c['compose_type'] = 'sum'
lm_config_registry['c1'] = c


# With no dict
c['dict_path'] = ''
lm_config_registry['c2'] = c

# With a dict only, without embeddings
c['dict_path'] = prefix + 'data/toy_data_1/dict.json'
c['disregard_word_embeddings'] = True
lm_config_registry['c3'] = c


#10K

# with dictionary
c = lm_config_registry['c1']
c['data_path'] = prefix + 'data/toy_data_1_1'
c['dict_path'] = prefix + 'data/toy_data_1_1/dict.json'
lm_config_registry['c4'] = c

# ... without dictionary
c['dict_path'] = ''
lm_config_registry['c5'] = c

# with dictionary, with fully connected compositio
c = lm_config_registry['c4']
c['compose_type'] = 'fully_connected'
lm_config_registry['c6'] = c

# only with dictionary
c = lm_config_registry['c4']
c['disregard_word_embeddings'] = True
lm_config_registry['c7'] = c

# lower LR for fully connected composition
c = lm_config_registry['c6']
c['learning_rate'] = 0.0001
lm_config_registry['c8'] = c

# lower LR for dictless LM
c = lm_config_registry['c5']
c['learning_rate'] = 0.0001
lm_config_registry['c9'] = c


### new sets of experiments:
# LR to 0.0003. Maybe tanh  saturated so 
# try ReLU activations
# lower LR for dictless LM
c = lm_config_registry['c4']
c['compose_type'] = 'fully_connected_relu'
c['learning_rate'] = 0.0003
lm_config_registry['FC_relu0003'] = c

c = lm_config_registry['c4']
c['learning_rate'] = 0.0003
c['compose_type'] = 'fully_connected_linear'
lm_config_registry['FC_linear0003'] = c

c = lm_config_registry['c4']
c['dict_path'] = ''
c['learning_rate'] = 0.0003
lm_config_registry['nodict0003'] = c

### new sets of experiments:
c = lm_config_registry['root']
c['standalone_def_lookup'] = True
c['standalone_def_rnn'] = True
c['learning_rate'] = 0.0003
c['data_path'] = prefix + 'data/toy_data_3'
c['dict_path'] = prefix + 'data/toy_data_3/dict.json'
c['batch_size'] = 16
c['batch_size_valid'] = 16
c['top_k_words'] = 100
# Here I forgot to put exclude top k but it shouldn't change a thing
# as the primes doesn't have any def
lm_config_registry['d_dict_std'] = c

c = lm_config_registry['d_dict_std']
c['dict_path'] = ''
lm_config_registry['d_no_dict'] = c

c = lm_config_registry['d_dict_std']
c['compose_type'] = 'fully_connected_linear'
lm_config_registry['d_dict_lin'] = c

c = lm_config_registry['d_dict_std']
c['compose_type'] = 'linear_and_sum'
lm_config_registry['d_dict_lin_sum'] = c

c = lm_config_registry['d_dict_lin']
c['dim'] = 32
lm_config_registry['d_dict_lin_sd'] = c

c = lm_config_registry['d_no_dict']
c['dim'] = 32
lm_config_registry['d_no_dict_sd'] = c

# New set of exps with toy_data_2 which is bigger than 3
c = lm_config_registry['root']
c['standalone_def_lookup'] = True
c['standalone_def_rnn'] = True
c['learning_rate'] = 0.0003
c['data_path'] = prefix + 'data/toy_data_2'
c['dict_path'] = prefix + 'data/toy_data_2/dict.json'
c['batch_size'] = 16
c['batch_size_valid'] = 16
c['top_k_words'] = 100
#c['exclude_top_k'] = 100
c['compose_type'] = 'sum'
c['dim'] = 32
lm_config_registry['d2_dict_sum_sd'] = c

c = lm_config_registry['d2_dict_sum_sd']
c['dict_path'] = ''
lm_config_registry['d2_no_dict_sd'] = c

