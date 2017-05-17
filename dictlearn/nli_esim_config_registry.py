from dictlearn.config_registry import ConfigRegistry

nli_esim_config_registry = ConfigRegistry()

# Each epoch has ~500k examples
# Params copied from https://github.com/NYU-MLL/multiNLI/blob/master/python/util/parameters.py
nli_esim_config_registry.set_root_config({
    'data_path':  'snli',
    'layout': 'snli',

    # Lookup params
    'max_def_per_word': 100000,
    'emb_dim': 300,
    'bn': 0,
    'dim': 300,
    'dict_path': '',
    'vocab': '',
    'vocab_text': '', # Defaults to vocab. Use when original vocab cannot be used for frequency in dict
    'encoder': 'bilstm',

    # Also used in NYU-MLI
    'embedding_path': '',
    'train_emb': 1,
    'train_def_emb': 1,

    # Dict params
    'vocab_def': '',
    'compose_type': '',
    'try_lowercase': True,
    'disregard_word_embeddings': False,
    'exclude_top_k': -1,
    'max_def_length': 50,
    'embedding_def_path': '',
    'def_dim': 100,
    'combiner_reader_translate': False,
    'def_emb_translate_dim': -1,
    'def_emb_dim': -1,
    "combiner_dropout": 0.0,
    'num_input_def_words': 0,
    "combiner_dropout_type": "per_unit",
    'with_too_long_defs': 'drop',
    "combiner_gating": "none",
    "combiner_shortcut": False,
    'reader_type': 'mean',
    'share_def_lookup': False,
    'combiner_bn': False,

    'num_input_words': -1, # Will take vocab size
    "dropout": 0.5,
    'batch_size': 32,
    'lr': 0.0004,

    # Misc. Monitor every 100% of epoch
    'monitor_parameters': 0,
    'mon_freq': int((500000) / 32) / 2, # 2 times per epoch
    'save_freq_epochs': 1,
    'mon_freq_valid': int((500000) / 32) / 2,
    'n_batches': 150 * (500000 / 32) # ~50 epochs of SNLI
})

assert nli_esim_config_registry['root']['batch_size'] == 32, "Correct n_batches"

### Establish baseline ###
c = nli_esim_config_registry['root']
nli_esim_config_registry['baseline'] = c

c = nli_esim_config_registry['root']
c['train_emb'] = 0
c['embedding_path'] = 'snli/glove.840B.300d.npy'
nli_esim_config_registry['baseline_glove'] = c

c = nli_esim_config_registry['root']
c['num_input_words'] = 3000
nli_esim_config_registry['baseline_3k'] = c

### Dict configs ###

# Simple dict
c = nli_esim_config_registry['root']
c['dict_path'] = 'snli/wordnet_dict_add_lower_lowerllemma.json'
c['num_input_def_words'] = 11000
c['combiner_shortcut'] = False
c['combiner_reader_translate'] = False
c['data_path'] = 'snli'
c['def_dim'] = 100
c['def_emb_dim'] = 100
c['emb_dim'] = 100
c['layout'] = 'snli'
c['reader_type'] = 'mean'
c['max_def_per_word'] = 20
c['embedding_def_path'] = ''
c['combiner_dropout'] = 0.0
c['with_too_long_defs'] = 'drop'
c['train_emb'] = 1
c['embedding_path'] = ''
c['num_input_words'] = 3000
c['compose_type'] = 'sum'
nli_esim_config_registry['sum_small_dict_simple'] = c

c = nli_esim_config_registry['root']
c['dict_path'] = 'snli/wordnet_dict_add_lower_lowerllemma.json'
c['vocab_def'] = 'snli/wordnet_dict_add_lower_lowerllemma_vocab.txt'
c['num_input_def_words'] = -1
c['combiner_shortcut'] = False
c['combiner_reader_translate'] = False
c['data_path'] = 'snli'
c['def_dim'] = 100
c['def_emb_dim'] = 100
c['emb_dim'] = 100
c['layout'] = 'snli'
c['share_def_lookup'] = False
c['reader_type'] = 'mean'
c['max_def_per_word'] = 20
c['combiner_dropout'] = 0.0
c['with_too_long_defs'] = 'drop'
c['train_emb'] = 1
c['embedding_path'] = ''
c['embedding_def_path'] = ''
c['num_input_words'] = 3000
c['compose_type'] = 'sum'
nli_esim_config_registry['sum_small_dict'] = c

#####################
### Paper configs ###
#####################

# A) "Normal" baselines

c = nli_esim_config_registry['root']
c['num_input_words'] = 3000
c['emb_dim'] = 100
nli_esim_config_registry['paper_baseline_3k'] = c

c = nli_esim_config_registry['root']
c['num_input_words'] = 5000
c['emb_dim'] = 100
nli_esim_config_registry['paper_baseline_5k'] = c

c = nli_esim_config_registry['root']
c['num_input_words'] = 10000
c['emb_dim'] = 100
nli_esim_config_registry['paper_baseline_10k'] = c

c = nli_esim_config_registry['baseline']
c['train_emb'] = 0
c['n_batches'] = 150 * (500000 / c['batch_size'])
c['translate_dim'] = 100
c['vocab'] = 'snli/vocab_all.txt'
c['num_input_words'] = -1
c['embedding_path'] = 'snli/glove.840B.300d_all.npy'
nli_esim_config_registry['paper_baseline_glove'] = c

# B) Dict without glove

## Less tuned fellow
c = nli_esim_config_registry['root']
c['emb_dim'] = 100
c['def_emb_dim'] = 100
c['def_dim'] = 100
c['translate_dim'] = 100
c['combiner_shortcut'] = True
c['combiner_reader_translate'] = False
c['dict_path'] = 'snli/wordnet_dict_add_lower_lemma.json'
c['vocab_def'] = 'snli/wordnet_dict_add_lower_lemma_vocab.txt'
c['exclude_top_k'] = 3000
c['share_def_lookup'] = False
c['reader_type'] = 'mean' # Tune?
c['max_def_per_word'] = 20
c['combiner_dropout'] = 0.0
c['embedding_path'] = ''
c['num_input_words'] = 5000
c['num_input_def_words'] = 11000
c['compose_type'] = 'sum'
c['n_batches'] = 150 * (500000 / c['batch_size'])
c['train_emb'] = 1
c['embedding_path'] = ''
c['embedding_def_path'] = ''
c['combiner_reader_translate'] = False
nli_esim_config_registry['paper_dict_simple'] = c

# C) "Dict" baselines: lowercase + spelling

c = nli_esim_config_registry['paper_dict_simple']
c['dict_path'] = 'snli/dict_all_spelling.json'
c['vocab_def'] = 'snli/dict_all_spelling_vocab.txt' # Otherwise chars are UNK
c['n_batches'] = 150 * (500000 / c['batch_size'])
c['reader_type'] = 'rnn' # As pointed out by Dima reader should be LSTM for spelling
nli_esim_config_registry['paper_baseline_spelling'] = c

c = nli_esim_config_registry['paper_baseline_spelling']
c['dict_path'] = 'snli/dict_all_only_lowercase.json'
c['n_batches'] = 150 * (500000 / c['batch_size'])
c['reader_type'] = 'mean'
nli_esim_config_registry['paper_baseline_lowercase'] = c

# D) Dict + glove

# Note: I pick here one specific instance of glove init. It doesn't work anyways, and
# it went closest to actual glove result

def transform_glove(c):
    c['embedding_path'] = "glove/glove_w_specials.npy"
    c['embedding_def_path'] = "glove/glove_w_specials.npy"
    c['vocab'] = 'glove/vocab.txt'
    c['vocab_def'] = 'glove/vocab.txt'
    c['vocab_text'] = 'snli/vocab.txt'
    c['train_emb'] = 0
    c['train_def_emb'] = 0
    c['combiner_reader_translate'] = True
    c['def_emb_dim'] = 300
    c['emb_dim'] = 300
    c['def_emb_translate_dim'] = 100
    c['def_dim'] = 100
    c['num_input_def_words'] = -1
    c['num_input_words'] = -1
    return c

nli_esim_config_registry['paper_dict_simple_glove'] = transform_glove(nli_esim_config_registry['paper_dict_simple'])
