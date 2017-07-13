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

#####################
### Paper configs ###
#####################

# A) "Normal" baselines

c = nli_esim_config_registry['root']
c['num_input_words'] = 3000
c['emb_dim'] = 300
nli_esim_config_registry['paper_baseline_3k'] = c

c = nli_esim_config_registry['root']
c['num_input_words'] = 5000
c['emb_dim'] = 300
nli_esim_config_registry['paper_baseline_5k'] = c

c = nli_esim_config_registry['root']
c['train_emb'] = 0
c['n_batches'] = 100 * (500000 / c['batch_size'])
c['translate_dim'] = 100
c['vocab'] = 'snli/vocab_all.txt'
c['num_input_words'] = -1
c['embedding_path'] = 'snli/glove.840B.300d_all.npy'
nli_esim_config_registry['paper_baseline_glove'] = c

## Less tuned fellow
c = nli_esim_config_registry['root']
c['emb_dim'] = 100
c['def_emb_dim'] = 100
c['def_dim'] = 100
c['translate_dim'] = 100
c['combiner_shortcut'] = True
c['combiner_reader_translate'] = False
c['dict_path'] = 'snli/wordnet_dict_add_lower_lemma_lowerlemma.json'
c['exclude_top_k'] = 1000
c['share_def_lookup'] = False
c['reader_type'] = 'mean' # Tune?
c['max_def_per_word'] = 20
c['combiner_dropout'] = 0.0
c['embedding_path'] = ''
c['num_input_words'] = 3000
c['num_input_def_words'] = 3000
c['compose_type'] = 'sum'
c['n_batches'] = 100 * (500000 / c['batch_size'])
c['train_emb'] = 1
c['embedding_path'] = ''
c['embedding_def_path'] = ''
c['combiner_reader_translate'] = False
nli_esim_config_registry['paper_dict_simple'] = c

## Tune fellow
c = nli_esim_config_registry['paper_dict_simple']
c['dropout'] = 0.7
c['dim'] = 100
c['num_input_def_words'] = 11000
c['vocab_def'] = 'snli/wordnet_dict_add_lower_lemma_lowerlemma_vocab.txt'
c['combiner_dropout'] = 0.5
nli_esim_config_registry['paper_dict_tuned'] = c

c = nli_esim_config_registry['paper_dict_tuned']
c['dict_path']  = 'snli/wordnet_dict_add_lower_lemma_lowerlemma_spelling.json'
c['vocab_def'] = 'snli/wordnet_dict_add_lower_lemma_lowerlemma_spelling_vocab.txt'
nli_esim_config_registry['paper_dict_tuned_spelling'] = c

c = nli_esim_config_registry['paper_dict_simple']
c['dict_path'] = 'snli/dict_all_spelling.json'
c['vocab_def'] = 'snli/dict_all_spelling_vocab.txt' # Otherwise chars are UNK
c['reader_type'] = 'rnn' # As pointed out by Dima reader should be LSTM for spelling
nli_esim_config_registry['paper_baseline_spelling'] = c

c = nli_esim_config_registry['paper_dict_tuned']
c['dict_path'] = 'snli/dict_all_spelling.json'
c['vocab_def'] = 'snli/dict_all_spelling_vocab.txt' # Otherwise chars are UNK
c['reader_type'] = 'rnn' # As pointed out by Dima reader should be LSTM for spelling
nli_esim_config_registry['paper_baseline_spelling_tuned'] = c

c = nli_esim_config_registry['paper_baseline_spelling_tuned']
c['dim'] = 300
nli_esim_config_registry['paper_baseline_spelling_tuned_dim=300'] = c

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
