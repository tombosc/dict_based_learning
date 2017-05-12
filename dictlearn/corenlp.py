"""Corenlp interface.

Copy-pasted from https://github.com/smilli/py-corenlp,
and unfortunately there was no license in that repository
at the time.

"""
import time
import json
import requests
import subprocess


def start_corenlp(port):
    """Make sure to set CLASSPATH"""
    command = "java -mx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -quiet -port {} -timeout 15000".format(port)
    popen = subprocess.Popen(command.split())
    # ugly!
    time.sleep(2.0)
    return popen


class StanfordCoreNLP(object):

    def __init__(self, server_url):
        self.server_url = server_url

    def annotate(self, text, properties=None):
        assert isinstance(text, unicode)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
            '$ cd stanford-corenlp-full-2015-12-09/ \n'
            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')

        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=text.encode('utf-8'), headers={'Connection': 'close'})
        output = r.text
        if ('outputFormat' in properties
             and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf-8', strict=True)
            except:
                pass
        return output


    def tokenize(self, str_):
        annotations = json.loads(
            self.annotate(str_,
                                properties={'annotators': 'tokenize,ssplit'}))
        tokens = []
        positions = []
        for sentence in annotations['sentences']:
            for token in sentence['tokens']:
                tokens.append(token['originalText'])
                positions.append(token['characterOffsetBegin'])
        return tokens, positions
