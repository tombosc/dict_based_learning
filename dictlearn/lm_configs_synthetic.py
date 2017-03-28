from dictlearn.config_registry import ConfigRegistry
from dictlearn.language_model_configs import lm_config_registry


# With dictionary
c = lm_config_registry['root']
c['data_path'] = '/data/lisa/exp/bosctom/dict_based_learning/data/toy_data_1'
c['dict_path'] = '/data/lisa/exp/bosctom/dict_based_learning/data/toy_data_1/dict.json'
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
c['dict_path'] = '/data/lisa/exp/bosctom/dict_based_learning/data/toy_data_1/dict.json'
c['disregard_word_embeddings'] = True
lm_config_registry['c3'] = c


#10K

# with dictionary
c = lm_config_registry['c1']
c['data_path'] = '/data/lisa/exp/bosctom/dict_based_learning/data/toy_data_1_1'
c['dict_path'] = '/data/lisa/exp/bosctom/dict_based_learning/data/toy_data_1_1/dict.json'
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
