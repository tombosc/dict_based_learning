#!/usr/bin/env python

import argparse
import logging

from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary
from dictlearn.corenlp import start_corenlp
from dictlearn.util import get_free_port

def main():
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Crawl definitions for a vocabulary")
    parser.add_argument("--api_key",
                        help="Wordnik API key to use")
    # NOTE(kudkudak): wordnik has useCanonical which tries to do stuff like Cats -> cat
    # but it doesn't really work well
    parser.add_argument("--just-lemmas", action="store_true",
                        help="Just use the lemmas as the definition")
    parser.add_argument("--just-lowercase", action="store_true",
                        help="Just lowercase as the definition")
    parser.add_argument("--add-lemma-defs", action="store_true",
                        help="Add definitions from lemmas")
    parser.add_argument("--add-lower-defs", action="store_true",
        help="Add definitions from lowercase")
    parser.add_argument("--add-lower-lemma-defs", action="store_true",
        help="Add definitions from lowercase version of word and lemmas")
    parser.add_argument("--add-dictname-to-defs", action="store_true",
        help="Adds dictionary name to definition")
    parser.add_argument("--identity", action="store_true",
                        help="Identity mapping dictionary")
    parser.add_argument("--spelling", action="store_true",
                        help="Spelling dictionary")
    parser.add_argument("--crawl-also-lowercase", default=False,
        help="If true will crawl also lower-cased version")
    parser.add_argument("--crawl-also-lemma", default=False,
        help="If true will crawl also lemma version")
    parser.add_argument("vocab", help="Vocabulary path")
    parser.add_argument("dict", help="Destination path for the dictionary")
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    dict_ = Dictionary(args.dict)

    if args.api_key:
        port = get_free_port()
        try:
            popen = start_corenlp(port)
            dict_.crawl_wordnik(
                    vocab, args.api_key, "http://localhost:{}".format(port),
                crawl_also_lowercase=args.crawl_also_lowercase,
                crawl_also_lemma=args.crawl_also_lemma)
        finally:
            if popen and popen.returncode is None:
                popen.kill()
    # NOTE(kudkudak): A bit ugly, but this covers case where we have Cats which do not get added lemmas
    # from cat without try_lower=True
    elif args.add_lemma_defs or args.add_lower_lemma_defs:
        dict_.add_from_lemma_definitions(vocab, try_lower=args.add_lower_lemma_defs)
    elif args.add_lower_defs:
        dict_.add_from_lowercase_definitions(vocab)
    elif args.add_dict_name_def:
        dict_.add_dictname_to_defs(vocab)
    elif args.just_lemmas:
        dict_.crawl_lemmas(vocab)
    elif args.just_lowercase:
        dict_.crawl_lowercase(vocab)
    elif args.identity:
        dict_.setup_identity_mapping(vocab)
    elif args.spelling:
        dict_.setup_spelling(vocab)
    else:
        raise ValueError("don't know what to do")

if __name__ == "__main__":
    main()
