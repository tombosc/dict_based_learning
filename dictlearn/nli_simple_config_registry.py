from dictlearn.config_registry import ConfigRegistry

snli_config_registry = ConfigRegistry()

snli_config_registry.set_root_config({
    'data_path':  'snli/',
    'layout': 'snli',

    # Lookup params
    'translate_dim': 300,
    'max_def_per_word': 100000,
    'bn': True,
    'mlp_dim': 600,
    'emb_dim': 300, # Used for def and word lookup
    'dict_path': '',

    # Remove by default embeddings. Our goal ATM is to beat random init
    'embedding_path': '', #/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.npy',

    'vocab_def': '',
    'vocab_text': '', # If passed will be used for exclude_top_k in Retrieval only
    'vocab': '',

    'def_dim': 300, # LSTM reader hidden state or translate in MeanPool
    'def_emb_dim': -1, # Dimensionality of vectors used in definitions
    'compose_type': '',
    'disregard_word_embeddings': False,
    'exclude_top_k': -1,
    'max_def_length': 50,
    'with_too_long_defs': 'drop',
    'train_emb': 1, # Remove by default embeddings. Our goal ATM is to beat random init
    "combiner_dropout": 0.0,
    "combiner_dropout_type": "per_unit",
    "combiner_gating": "none",
    "combiner_reader_translate": True,
    "combiner_shortcut": False,
    'reader_type': 'rnn',
    'share_def_lookup': False,
    'combiner_bn': False,

    'num_input_words': 0, # Will take vocab size
    'num_input_def_words': 0, # Will take vocab size

    "encoder": "sum",
    "dropout": 0.3,
    'batch_size': 512,
    'lr': 0.001,
    'l2': 4e-6,

    # Misc
    'monitor_parameters': 0,
    'mon_freq': 1000,
    'save_freq_epochs': 1,
    'mon_freq_valid': 1000,
    'n_batches': 200000 # ~200 epochs of SNLI with batch size 500 (!
})

### Establish baselines ###
c = snli_config_registry['root']
snli_config_registry['baseline'] = c

c = snli_config_registry['baseline']
c['train_emb'] = 0 # Following Squad convention
c['embedding_path'] = 'snli/glove.840B.300d.npy'
snli_config_registry['baseline_glove'] = c

c = snli_config_registry['root']
c['data_path'] = 'mnli/'
c['layout'] = 'mnli'
snli_config_registry['baseline_mnli'] = c

c = snli_config_registry['root']
c['train_emb'] = 0 # Following Squad convention
c['embedding_path'] = 'mnli/glove.840B.300d.npy'
c['data_path'] = 'mnli/'
c['layout'] = 'mnli'
snli_config_registry['baseline_mnli_glove'] = c

### Small dict ###

c = snli_config_registry['root']
c['dict_path'] = 'mnli/dict_all_with_lowercase.json'
c['data_path'] = 'mnli/'
c['layout'] = 'mnli'
c['exclude_top_k'] = 2500
c['share_def_lookup'] = False
c['reader_type'] = 'mean'
c['max_def_per_word'] = 20 # Allows to have strict calculation on memory consumption
c['combiner_dropout'] = 0.0
c['train_emb'] = 1
c['embedding_path'] = ''
c['num_input_words'] = 3000
c['num_input_def_words'] = 3000
c['compose_type'] = 'sum'
snli_config_registry['sum_small_dict_mnli'] = c

c = snli_config_registry['root']
c['dict_path'] = 'snli/dict_all.json'
c['data_path'] = 'snli/'
c['exclude_top_k'] = 2500
c['share_def_lookup'] = False
c['layout'] = 'snli'
c['reader_type'] = 'mean'
c['max_def_per_word'] = 20
c['combiner_dropout'] = 0.0
c['train_emb'] = 1
c['embedding_path'] = ''
c['num_input_words'] = 3000
c['num_input_def_words'] = 3000
c['compose_type'] = 'sum'
snli_config_registry['sum_small_dict'] = c

#####################
### Paper configs ###
#####################

# A) "Normal" baselines

c = snli_config_registry['root']
c['num_input_words'] = 3000
c['emb_dim'] = 100
snli_config_registry['paper_baseline_3k'] = c

c = snli_config_registry['root']
c['num_input_words'] = 5000
c['emb_dim'] = 100
snli_config_registry['paper_baseline_5k'] = c

c = snli_config_registry['baseline']
c['train_emb'] = 0
c['n_batches'] = 100000
c['translate_dim'] = 100
c['vocab'] = 'snli/vocab_all.txt'
c['num_input_words'] = -1
c['embedding_path'] = 'snli/glove.840B.300d_all.npy'
snli_config_registry['paper_baseline_glove'] = c

# B) Dict without glove

## Less tuned fellow
## Should be same as results_snli_2_05_sum_small_dict_3k_exclude1k_shortcut_transl_new_dict_all
c = snli_config_registry['root']
c['batch_size'] = 512
c['dropout'] = 0.3
c['emb_dim'] = 300
c['def_emb_dim'] = 300
c['def_dim'] = 300
c['batch_size'] = 512
c['combiner_shortcut'] = True
c['dict_path'] = 'snli/dict_all_3_05_lowercase_lemma.json'
c['data_path'] = 'snli/'
c['exclude_top_k'] = 1000
c['combiner_shortcut'] = True
c['share_def_lookup'] = False
c['layout'] = 'snli'
c['reader_type'] = 'mean'
c['max_def_per_word'] = 20
c['combiner_dropout'] = 0.0
c['translate_dim'] = 300
c['train_emb'] = 1
c['embedding_path'] = ''
c['num_input_words'] = 3000
c['num_input_def_words'] = 3000
c['compose_type'] = 'sum'
c['n_batches'] = 200000
c['combiner_reader_translate'] = False
snli_config_registry['paper_dict_simple'] = c

## More tuned fellow
## Should be same as results_snli_8_05_sum_small_dict_3k_exclude1k_embdim=100_udrop=0.2_bs256_noreg
c = snli_config_registry['paper_dict_simple']
c['def_dim'] = 100
c['def_emb_dim'] = 100
c['emb_dim'] = 100
c['translate_dim'] = 300
c['batch_size'] = 256
c['combiner_dropout'] = 0.2
c['combiner_dropout_type'] = 'per_unit'
c['num_input_def_words'] = 11000
c['vocab_def'] = 'snli/dict_all_3_05_lowercase_lemma_vocab.txt'
c['dropout'] = 0.15
c['l2'] = 0.0
c['combiner_reader_translate'] = False
snli_config_registry['paper_dict_tuned'] = c

# C) "Dict" baselines: lowercase + spelling

c = snli_config_registry['paper_dict_simple']
c['dict_path'] = 'snli/dict_all_spelling.json'
c['vocab_def'] = 'snli/dict_all_spelling_vocab.txt' # Otherwise chars are UNK
c['n_batches'] = 100000
c['reader_type'] = 'rnn' # As pointed out by Dima reader should be LSTM for spelling
snli_config_registry['paper_baseline_spelling'] = c

c = snli_config_registry['paper_baseline_spelling']
c['dict_path'] = 'snli/dict_all_only_lowercase.json'
c['n_batches'] = 100000
c['reader_type'] = 'mean'
snli_config_registry['paper_baseline_lowercase'] = c

# D) Dict + glove

# Note: I pick here one specific instance of glove init. It doesn't work anyways, and
# it went closest to actual glove result

def transform_glove(c):
    c['embedding_path'] = "glove/glove_w_specials.npy"
    c['embedding_def_path'] = "glove/glove_w_specials.npy"
    c['vocab'] = 'glove/vocab.txt'
    c['vocab_def'] = 'glove/vocab.txt'
    c['vocab_text'] = 'data/vocab.txt'
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

snli_config_registry['paper_dict_simple_glove'] = transform_glove(snli_config_registry['paper_dict_simple'])
snli_config_registry['paper_dict_tuned_glove'] = transform_glove(snli_config_registry['paper_dict_tuned'])
