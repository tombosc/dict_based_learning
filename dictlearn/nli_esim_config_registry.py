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
    'encoder': 'bilstm',

    # Also used in NYU-MLI
    'embedding_path': '',
    'train_emb': 1,

    # Dict params
    'vocab_def': '',
    'compose_type': '',
    'try_lowercase': True,
    'disregard_word_embeddings': False,
    'exclude_top_k': -1,
    'max_def_length': 50,
    'def_dim': 100,
    'def_emb_dim': -1,
    "combiner_dropout": 0.0,
    "combiner_dropout_type": "per_unit",
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
    'mon_freq': int((500000) / 32) / 3, # 3 times per epoch
    'save_freq_epochs': 1,
    'mon_freq_valid': int((500000) / 32) / 3,
    'n_batches': 200 * (500000 / 32) # ~200 epochs of SNLI
})

### Establish baseline ###
c = nli_esim_config_registry['root']
nli_esim_config_registry['baseline'] = c

c = nli_esim_config_registry['root']
c['train_emb'] = 0
c['embedding_path'] = 'data/snli/glove.840B.300d.npy'
nli_esim_config_registry['baseline_glove'] = c

c = nli_esim_config_registry['root']
c['num_input_words'] = 3000
nli_esim_config_registry['baseline_3k'] = c

### Dict configs ###

# Looking up words from test/dev as well

"""
python bin/train_nli_baseline.py sum_small_dict
results/snli/8_05_sum_small_dict_10k_exclude1k_shortcut_transl_new_dict_all_defcoab11k_dim=100_bs256_noreg_glove
--emb_dim=300 --batch_size=256 --def_emb_dim=100 --no_combiner_reader_translate --translate_dim=100 --def_dim=100
--dict_path=data/snli/dict_all_3_05_lowercase_lemma_add_all.json --num_input_def_words=11000
--vocab_def=data/snli/dict_all_3_05_lowercase_lemma_vocab.txt --combiner_shortcut --exclude_top_k=1000
--num_input_words=3000 --n_batches=200000 --dropout=0.15 --l2=0
--embedding_path=/data/lisa/exp/jastrzes/dict_based_learning/data/snli/glove.840B.300d.npy --train_emb=0 --emb_dim=300
"""
#
# c = nli_esim_config_registry['root']
# c['dict_path'] = 'data/snli/dict_all_with_lowercase.json'
# c['data_path'] = 'snli'
# c['layout'] = 'snli'
# c['exclude_top_k'] = 1000
# c['share_def_lookup'] = False
# c['reader_type'] = 'mean'
# c['max_def_per_word'] = 20 # Allows to have strict calculation on memory consumption
# c['combiner_dropout'] = 0.0
# c['train_emb'] = 1
# c['embedding_path'] = ''
# c['num_input_words'] = 3000
# c['num_input_def_words'] = 3000
# c['compose_type'] = 'sum'
# snli_config_registry['sum_small_dict_mnli'] = c