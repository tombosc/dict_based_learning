from dictlearn.config_registry import ConfigRegistry

lm_config_registry = ConfigRegistry()
lm_config_registry.set_root_config({
    # data
    'data_path': 'onebillionword/',
    'dict_path' : "",
    'vocab_path': "",
    'dict_vocab_path': "",
    'layout' : 'standard',
    'num_input_words' : 10000,
    'def_num_input_words' : 0, #0 => num_input_words
    'num_output_words': 10000,
    'max_length' : 100,
    'batch_size' : 64,
    'batch_size_valid' : 64,
    'max_def_length' : 100,
    'max_def_per_word': 1000,
    'exclude_top_k' : 0,

    # model
    'emb_dim' : 500,
    'emb_def_dim': 500,
    'dim' : 500,
    'compose_type' : 'sum',
    'disregard_word_embeddings' : False,
    'learning_rate' : 0.001,
    'momentum' : 0.9,
    'grad_clip_threshold' : 5.0,

    # embeddings
    'embedding_path': '',

    # model: def_reader
    'def_reader': 'LSTM',
    'standalone_def_rnn' : False,
    'standalone_def_lookup': False,

    # monitoring and checkpointing
    'mon_freq_train' : 100,
    'mon_freq_valid' : 1000,
    'save_freq_batches' : 1000,
    'very_rare_threshold': 170000, # less than 1% of occurences are ranked above
    'n_batches' : 0,
    'monitor_parameters' : False,
    'fast_checkpoint' : False
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
c['max_def_length'] = 100
c['dict_path'] = 'dict_snli.json'
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

c['dict_path'] = 'dict_obw_reordered_max1_si.json'
lm_config_registry['obw_10k_dict_mean_si1'] = c

c = lm_config_registry['obw_10k_dict_mean']
c['max_def_per_word'] = 1 
lm_config_registry['obw_10k_dict_mean_r1'] = c

c = lm_config_registry['obw_10k_dict3_n']
c['dict_path'] = 'dict_obw_2.json'
c['max_def_per_word'] = 3
lm_config_registry['obw_10k_dict3_n2'] = c

c = lm_config_registry['obw_10k_dict1']
c['dict_path'] = 'dict_obw.json'
c['max_def_per_word'] = 3
lm_config_registry['obw_10k_dict1_n'] = c

c = lm_config_registry['obw_10k_dict2']
c['dict_path'] = 'dict_obw.json'
c['max_def_per_word'] = 3
lm_config_registry['obw_10k_dict2_n'] = c

def new_dictify(c):
    c['dict_path'] = 'onebillionword/wn/dict_wn.json'
    c['max_def_per_word'] = 14
    c['max_def_length'] = 30
    return c

c = lm_config_registry['obw_10k_dict1']
c = new_dictify(c)
lm_config_registry['obw_10k_dict1_wn'] = c
c = lm_config_registry['obw_10k_dict2']
c = new_dictify(c)
lm_config_registry['obw_10k_dict2_wn'] = c

# BEGIN OLD GLOVE
# these use same embedding size for GloVe and trainable which is a problem
# to compare with baselines which use emb_dim=500
# also they use huge vocab and thus huge embedding matrix, that's why they are 
# restricted to 300k
# the 2 first are not trainable (i.e. there isn't a core of 10k trainable 
# embeddings, they only use the frozen embeddings)
# They used old implem where the dict was generated on the fly, so can't run 
# them anymore
c = lm_config_registry['obw_base_10k_slower']
c['embedding_path']= 'onebillionword/glove.840B.300d.300005.npy'
c['emb_def_dim'] = 300
c['emb_dim'] = 300
lm_config_registry['obw_base_10k_glove300k'] = c

c['num_input_words'] = 300005
lm_config_registry['obw_base_300k_glove300k'] = c

c = lm_config_registry['obw_base_10k_slower']
c['embedding_path']= 'onebillionword/glove.840B.300d.300005.npy'
c['dict_path'] = 'onebillionword/dict_obw_identity_300k.json'
c['emb_dim'] = 300
c['emb_def_dim'] = 300
c['exclude_top_k'] = 10000
c['def_reader'] = 'mean'
c['standalone_def_lookup'] = True
c['compose_type'] = 'transform_and_sum'
lm_config_registry['obw_base_10k_tr_glove_300k'] = c

c['embedding_path']= 'onebillionword/wn/glove.840B.300d_300k_common.npy'
lm_config_registry['obw_base_10k_tr_glove_300k_restr'] = c

# The following is a baseline for these (emb_dim=300)
c = lm_config_registry['obw_base_10k_slower']
c['emb_dim'] = 300
lm_config_registry['obw_base_10k_emb300'] = c

# This one is basically 30k trainable restricted to vocab that's also in glove 
# in the 0-30 range... 
# TODO(tombosc): double check and delete
c = lm_config_registry['obw_base_10k_slower']
c['vocab_path'] = 'onebillionword/wn/vocab_restricted_wnlemma_10k.txt'
c['num_input_words'] = 30000
lm_config_registry['obw_base_10k_restr_wnl_30k'] = c

# END OLD GLOVE

def new_dictify_addlemma(c):
    c['dict_path'] = 'onebillionword/wn/dict_obw_wn.json'
    c['max_def_per_word'] = 10 # covers > 95% (of defs, not occurences)
    c['max_def_length'] = 30 # covers > 99%
    return c

c = lm_config_registry['obw_10k_dict1']
c = new_dictify_addlemma(c)
lm_config_registry['obw_10k_dict1_wnl'] = c

c = lm_config_registry['obw_10k_dict2']
c = new_dictify_addlemma(c)
lm_config_registry['obw_10k_dict2_wnl'] = c

# Lemma lowercase restricted to 300k
c = lm_config_registry['obw_base_10k_slower']
c['dict_path'] = 'onebillionword/dict_identity_lemma_lowercase_300k.json'
c['def_reader'] = 'mean'
c['num_input_words'] = 10000
c['compose_type'] = 'transform_and_sum'
c['standalone_def_lookup'] = False
c['exclude_top_k'] = 10000
lm_config_registry['obw_base_10k_lemma_lc'] = c

c['dict_path'] = 'onebillionword/dict_obw_identity_lemma_lc.json'
lm_config_registry['obw_base_10k_lemma_lcf'] = c

c = lm_config_registry['obw_base_10k_slower']
c['num_input_words'] = 100
c['num_output_words']= 25
c['batch_size'] = 1
c['batch_size_valid'] = 1
# model
c['emb_dim'] = 7
c['dim'] = 9
c['compose_type'] = 'sum'
# monitoring and checkpointing
c['mon_freq_train'] = 1
c['mon_freq_valid'] = 100000
c['save_freq_batches'] = 1000
lm_config_registry['debug_1'] = c

c['batch_size'] = 32
c['batch_size_valid'] = 32
c['mon_freq_train'] = 10
c['mon_freq_valid'] = 100
c['save_freq_batches'] = 100
lm_config_registry['debug_2'] = c

c['emb_def_dim'] = 3
lm_config_registry['debug_3'] = c

c = lm_config_registry['obw_10k_dict2_wnl']
c['max_def_per_word'] = 20 # covers > 99% of the defs
lm_config_registry['obw_10k_dict2_wnl2'] = c

# NEW GLOVE
c = lm_config_registry['obw_base_10k_slower']
c['vocab_path'] = 'onebillionword/vocab_glove_10k.txt'
c['embedding_path']= 'onebillionword/glove.840B.300d.fitvocab10k.npy'
c['fast_checkpoint'] = True
c['dict_path'] = 'onebillionword/dict_obw_identity.json'
c['emb_dim'] = 500
c['emb_def_dim'] = 300
c['exclude_top_k'] = 10000
c['def_reader'] = 'mean'
c['standalone_def_lookup'] = True
c['compose_type'] = 'transform_and_sum'
c['def_num_input_words'] = 803808
lm_config_registry['10k_glove_lin'] = c

c = lm_config_registry['obw_base_10k_slower']
c['emb_dim'] = 500
c['emb_def_dim'] = 500
c['dim'] = 500
c['vocab_path'] = 'onebillionword/wn/vocab_restricted_wnlemma_10k_wm.txt'
c['num_input_words'] = 199312
lm_config_registry['train_where_def'] = c

c = lm_config_registry['10k_glove_lin']
c['vocab_path'] = 'onebillionword/vocab_glove_10k_restr_wnll.txt'
c['embedding_path'] = 'onebillionword/glove.840B.300d.10k_restr_wnll.npy'
c['def_num_input_words'] = 184045
lm_config_registry['10k_glove_where_def'] = c

c = lm_config_registry['obw_base_10k_slower']
c['standalone_def_lookup'] = True
c['standalone_def_rnn'] = True
c['fast_checkpoint'] = True
c['def_reader'] = 'LSTM'
c['compose_type'] = 'transform_and_sum'
c['exclude_top_k'] = 10000
c['vocab_path'] = '' # vocab.txt
c['def_num_input_words'] = 1000
c['dict_path'] = 'onebillionword/dict_obw_spelling_cut11.json' #max_def_len=11
c['dict_vocab_path'] = 'onebillionword/vocab_spelling_dict_weighted.txt'
lm_config_registry['10k_spelling'] = c

# we will augment dict2 with a separate lookup and a separate dict_vocab
c = lm_config_registry['obw_10k_dict2_wnl']
c['dict_vocab_path'] = 'onebillionword/wn/vocab_obw_wn_weighted.txt' 
c['def_num_input_words'] = 10000
c['standalone_def_lookup'] = True
c['standalone_def_rnn'] = True
c['fast_checkpoint'] = True
c['max_def_per_word'] = 20
lm_config_registry['obw_10k_dict5_wn'] = c
