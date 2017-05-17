FROM ubuntu:14.04
MAINTAINER Dzmitry Bahdanau <bahdanau@iro.umontreal.ca>
WORKDIR /workspace
RUN apt-get -y update
RUN apt-get -y install python-pip git
RUN apt-get -y install wget
RUN apt-get -y install python-dev
RUN apt-get -y install libhdf5-dev
RUN pip install numpy
RUN pip install h5py
RUN apt-get -y install libblas-dev
RUN apt-get -y install liblapack-dev

# ugly
RUN pip install -U six
RUN pip install numexpr
RUN pip install pillow
RUN pip install python_utils

# scipy
RUN apt-get install -y gfortran
RUN pip install scipy

# nltk
RUN pip install nltk
RUN export NLTK_DATA=/workspace/nltk_data
RUN python -m nltk.downloader wordnet -d /workspace/nltk_data
RUN python -m nltk.downloader perluniprops -d /workspace/nltk_data

# corenlp
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get -y update
RUN apt-get install -y openjdk-8-jre
RUN wget -q http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip 
RUN apt-get -y install unzip
RUN unzip stanford-corenlp-full-2016-10-31.zip

# wordnik (not necessary anymore)
RUN pip install wordnik

RUN git clone https://github.com/Theano/Theano
RUN cd Theano; pip install .; cd ../
RUN git clone https://github.com/mila-udem/fuel
RUN cd fuel; pip install .; cd ../
RUN git clone https://github.com/mila-udem/blocks
RUN cd blocks; pip install .; cd ../

ARG CACHEBUST=1
RUN git clone https://github.com/tombosc/dict_based_learning.git
RUN pip install -r dict_based_learning/requirements.txt


