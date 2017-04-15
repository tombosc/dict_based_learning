import json
import logging
import traceback
import h5py

from fuel.datasets.hdf5 import H5PYDataset

from dictlearn.corenlp import StanfordCoreNLP

logger = logging.getLogger()


def _find_sublist(list_, sublist):
    indices = []
    for i in range(len(list_) - len(sublist) + 1):
        found = True
        for j in range(len(sublist)):
            if list_[i + j] != sublist[j]:
                found = False
                break
        if found:
            indices.append(i)
    return indices


def text_to_h5py_dataset(text_path, dst_path):
    # The simplest is to load everything to memory first.
    # If memory becomes an issue, this code can be optimized.
    words = []
    with open(text_path, 'r') as src:
        for line in src:
            words.extend(line.strip().split())

    with h5py.File(dst_path, 'w') as dst:
        dtype = h5py.special_dtype(vlen=bytes)
        table = dst.create_dataset('words', (len(words),), dtype=dtype)
        table[:] = words

        dst.attrs['split'] = H5PYDataset.create_split_array({
                'train' : {
                    'words' : (0, len(words))
                }
            })


def squad_to_h5py_dataset(squad_path, dst_path, corenlp_url):
    data = json.load(open(squad_path))
    data = data['data']

    text = []
    def add_text(list_):
        text.extend(list_)
        return len(text) - len(list_), len(text)

    corenlp = StanfordCoreNLP(corenlp_url)
    def tokenize(str_):
        annotations = json.loads(
            corenlp.annotate(str_,
                             properties={'annotators': 'tokenize,ssplit'}))
        tokens = []
        positions = []
        for sentence in annotations['sentences']:
            for token in sentence['tokens']:
                tokens.append(token['originalText'])
                positions.append(token['characterOffsetBegin'])
        return tokens, positions


    all_contexts = []
    all_questions = []
    all_answer_begins = []
    all_answer_ends = []

    num_issues = 0
    for article in data:
        for paragraph in article['paragraphs']:
            context, context_positions = tokenize(paragraph['context'])
            context_begin, context_end = add_text(context)

            for qa in paragraph['qas']:
                try:
                    question, _ = tokenize(qa['question'])
                    question_begin, question_end = add_text(question)
                    answer_begins = []
                    answer_ends = []

                    for answer in qa['answers']:
                        start = answer['answer_start']
                        assert paragraph['context'][start:start + len(answer['text'])] == answer['text']
                        answer_text, _ = tokenize(answer['text'])
                        begin = context_positions.index(answer['answer_start'])

                        end = begin + len(answer_text)
                        answer_begins.append(begin)
                        answer_ends.append(end)

                    all_contexts.append((context_begin, context_end))
                    all_questions.append((question_begin, question_end))
                    all_answer_begins.append(answer_begins)
                    all_answer_ends.append(answer_ends)
                except ValueError:
                    logger.error("tokenized context: {}".format(zip(context, context_positions)))
                    logger.error("qa: {}".format(qa))
                    traceback.print_exc()
                    num_issues += 1
    if num_issues:
        logger.error("there were {} issues".format(num_issues))

    num_examples = len(all_contexts)

    dst = h5py.File(dst_path, 'w')
    unicode_dtype = h5py.special_dtype(vlen=unicode)
    dst.create_dataset('text', (len(text),), unicode_dtype)
    dst.create_dataset('contexts', (num_examples, 2), 'int64')
    dst.create_dataset('questions', (num_examples, 2), 'int64')
    vlen_int64 = h5py.special_dtype(vlen='int64')
    dst.create_dataset('answer_begins', (num_examples,), vlen_int64)
    dst.create_dataset('answer_ends', (num_examples,), vlen_int64)
    dst['text'][:] = text
    dst['contexts'][:] = all_contexts
    dst['questions'][:] = all_questions
    dst['answer_begins'][:] = all_answer_begins
    dst['answer_ends'][:] = all_answer_ends
    dst.attrs['split'] = H5PYDataset.create_split_array({
            'all' : {
                'contexts' : (0, num_examples),
                'questions' : (0, num_examples),
                'answer_begins' : (0, num_examples),
                'answer_ends' : (0, num_examples)
            }
        })
    dst.close()


def add_words_ids_to_squad(h5_file, vocab):
    """Digitizes test with a vocabulary.

    Also saves the vocabulary into the hdf5 file.

    """
    with h5py.File(h5_file, 'a') as dst:
        unicode_dtype = h5py.special_dtype(vlen=unicode)
        dst.create_dataset('text_ids', (dst['text'].shape[0],), 'int64')
        dst.create_dataset('vocab_words', (vocab.size(),), unicode_dtype)
        dst.create_dataset('vocab_freqs', (vocab.size(),), 'int64')
        dst['text_ids'][:] = map(vocab.word_to_id, dst['text'][:])
        dst['vocab_words'][:] = vocab.words
        dst['vocab_freqs'][:] = vocab.frequencies
