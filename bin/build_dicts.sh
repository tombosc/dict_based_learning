#!/usr/bin/env bash

set -x

[ -z $CLASSPATH ] && export CLASSPATH=$HOME/Dist/stanford-corenlp-full-2016-10-31/* 
[ -z $DBL ] && DBL=$HOME/Dist/dict_based_learning

$DBL/bin/build_vocab.py train.h5,dev.h5 vocab_combined.txt

# Builds a basic Wordnet dictionary.
# It should not matter which vocab is used.
# $DBL/bin/crawl_dict.py --wordnet vocab_combined.txt dict_wordnet.json 2>log_dict_wordnet.txt

rsync --progress -L dict_wordnet.json dict_wordnet2.2.json
$DBL/bin/crawl_dict.py --add-lower-lemma-defs vocab_combined.txt dict_wordnet2.2.json

# This will have almost no effect, as in most cases --add-lower-lemma-defs will add 
# the definition from the lowercased version of the word. However, sometimes the lemmatizer crashes,
# and for words on which it crashed this will work.
rsync --progress -L dict_wordnet2.2.json dict_wordnet3.2.json
$DBL/bin/crawl_dict.py --add-lower-defs vocab_combined.txt dict_wordnet3.2.json

# This wan't really used in any important experiments, but maybe in the future we will return to this.
rsync --progress -L dict_wordnet3.2.json dict_wordnet3.3.json
 $DBL/bin/crawl_dict.py --add-identity vocab_combined.txt dict_wordnet3.3.json

rsync --progress -L dict_wordnet3.2.json dict_wordnet4.3.json
$DBL/bin/crawl_dict.py --add-spelling vocab_combined.txt dict_wordnet4.3.json

# This is just a fancy way of finding 100 most frequent characters
$DBL/bin/build_vocab.py --exclude-top-k=3000 --weight-dict-entries vocab.txt dict_wordnet4.3.json vocab_dict_wordnet4.3.txt
grep '#' vocab_dict_wordnet4.3.txt | head -100 >100_most_frequent_chars.txt

# Build the vocabulary of characters
head -5 vocab.txt >vocab_chars.txt
cat 100_most_frequent_chars.txt >>vocab_chars.txt

# Build the mixed vocabulary
head -5 vocab.txt >vocab_with_chars.txt
cat 100_most_frequent_chars.txt >>vocab_with_chars.txt
awk 'NR > 5 { print }' vocab.txt >>vocab_with_chars.txt

# a baseline lemmatization dict
$DBL/bin/crawl_dict.py --identity vocab_combined.txt dict_identity.json
rsync --progress -L dict_identity.json dict_lemmalower.json
$DBL/bin/crawl_dict.py --add-lower-lemma-defs vocab_combined.txt dict_lemmalower.json

# Build spelling and spelling+lemmatization dictionaries
$DBL/bin/crawl_dict.py --add-spelling vocab_combined.txt dict_spelling2.1.json
rsync --progress -L dict_spelling2.1.json dict_spelling3.1.json
$DBL/bin/crawl_dict.py --add-lower-lemma-def vocab_combined.txt dict_spelling3.1.json

# A new generation of dictionaries that I have not yet properly tried
rsync dict_wordnet.json dict_wordnet5.0.json
$DBL/bin/crawl_dict.py --add-identity vocab_combined.txt dict_wordnet5.0.json
$DBL/bin/crawl_dict.py --add-spelling vocab_combined.txt dict_wordnet5.0.json
$DBL/bin/crawl_dict.py --add-lower-lemma-def vocab_combined.txt dict_wordnet5.0.json

$DBL/bin/crawl_dict.py --add-identity vocab_combined.txt dict_spelling4.0.json
$DBL/bin/crawl_dict.py --add-spelling vocab_combined.txt dict_spelling4.0.json
$DBL/bin/crawl_dict.py --add-lower-lemma-def vocab_combined.txt dict_spelling4.0.json
