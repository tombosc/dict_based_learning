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
    parser.add_argument("--wordnet", action="store_true",
        help="Crawl WordNet")
    parser.add_argument("--identity", action="store_true",
                        help="Identity mapping dictionary")
    parser.add_argument("--add-spelling-if-no-def", action="store_true",
                        help="Add spelling if there is no definition")
    parser.add_argument("--add-spelling", action="store_true",
                        help="Always add spelling")
    parser.add_argument("--crawl-also-lowercase", default=0, type=int,
        help="If true will crawl also lower-cased version")
    parser.add_argument("--crawl-also-lemma", default=0, type=int,
        help="If true will crawl also lemma version")
    parser.add_argument("vocab", help="Vocabulary path")
    parser.add_argument("dict", help="Destination path for the dictionary")
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    dict_ = Dictionary(args.dict)

    try:
        port = get_free_port()
        popen = start_corenlp(port)
        url = "http://localhost:{}".format(port)
        if args.api_key:
            dict_.crawl_wordnik(
                vocab, args.api_key, url,
                crawl_also_lowercase=args.crawl_also_lowercase,
                crawl_also_lemma=args.crawl_also_lemma)
        elif args.wordnet:
            dict_.crawl_wordnet(url)
        elif args.add_lemma_defs or args.add_lower_lemma_defs:
            # NOTE(kudkudak): A bit ugly, but this covers case where
            # we have Cats which do not get added lemmas
            # from cat without try_lower=True
            dict_.add_from_lemma_definitions(vocab, try_lower=args.add_lower_lemma_defs)
        elif args.add_lower_defs:
            dict_.add_from_lowercase_definitions(vocab)
        elif args.add_dictname_to_defs:
            dict_.add_dictname_to_defs(vocab)
        elif args.add_spelling_if_no_def:
            dict_.add_spelling(vocab)
        elif args.add_spelling:
            dict_.add_spelling(vocab, only_if_no_def=False)
        elif args.just_lemmas:
            dict_.crawl_lemmas(vocab)
        elif args.just_lowercase:
            dict_.crawl_lowercase(vocab)
        elif args.identity:
            dict_.setup_identity_mapping(vocab)
        else:
            raise ValueError("don't know what to do")
    finally:
        if popen and popen.returncode is None:
            popen.kill()

if __name__ == "__main__":
    main()
