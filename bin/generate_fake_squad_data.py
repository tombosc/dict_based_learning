#!/usr/bin/env python

import argparse
import logging

import numpy
import json

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Generate fake data in SQuAD format")
    parser.add_argument("--sentences", type=int, help="Number of sentences to use")
    parser.add_argument("txt", help="Text data to use")
    parser.add_argument("dst", help="Destination for fake squad data")
    args = parser.parse_args()

    rng = numpy.random.RandomState(1)

    src = open(args.txt)
    articles = []
    for line in src:
        line = line.strip().decode('utf-8')
        words = line.split()
        if len(words) < 10:
            continue
        if len(articles) == args.sentences:
            break
        qas = []
        for i in range(10):
            q_len = rng.randint(2, 5)
            q_begin = rng.randint(0, len(words) - q_len - 2)
            q_end = q_begin + q_len
            question = u" ".join(words[q_begin:q_end])
            answer = u" ".join(words[q_end:q_end + 2])
            answer_start = q_end + sum(map(len, words[:q_end]))
            qas.append({'question': question,
                        'answers': [{'text': answer, 'answer_start': answer_start}]})
        articles.append({'paragraphs': [{'context': line, 'qas': qas}]})

    with open(args.dst, 'w') as dst:
        json.dump({'data': articles}, dst, indent=2)


if __name__ == "__main__":
    main()
