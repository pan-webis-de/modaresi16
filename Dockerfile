FROM pasmod/miniconder2

RUN apt-get update && \
	apt-get install -y build-essential libxml2-dev libxslt-dev python-matplotlib libsm6 libxrender1 libfontconfig1 libicu-dev python-dev libhunspell-dev && \
	apt-get clean

# install packages with conda
RUN conda install -y \
  pip \
  numpy \
  pandas \
  scikit-learn \
  nltk \
  h5py \
  matplotlib

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')" 

WORKDIR /var/www
ADD . .
RUN pip install -r requirements.txt
RUN pip install -e .

RUN polyglot download --dir /root/polyglot_data pos2.nl
RUN polyglot download --dir /root/polyglot_data embeddings2.nl
RUN polyglot download --dir /root/polyglot_data pos2.en
RUN polyglot download --dir /root/polyglot_data embeddings2.en
RUN polyglot download --dir /root/polyglot_data pos2.es
RUN polyglot download --dir /root/polyglot_data embeddings2.es

RUN mkdir -p /root/hunspell
ADD https://cgit.freedesktop.org/libreoffice/dictionaries/plain/nl_NL/nl_NL.aff  /root/hunspell/nl_NL.aff
ADD https://cgit.freedesktop.org/libreoffice/dictionaries/plain/nl_NL/nl_NL.dic /root/hunspell/nl_NL.dic
ADD https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.aff /root/hunspell/en_US.aff
ADD https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic /root/hunspell/en_US.dic
ADD https://cgit.freedesktop.org/libreoffice/dictionaries/plain/es/es_ANY.dic /root/hunspell/es_ANY.dic
ADD https://cgit.freedesktop.org/libreoffice/dictionaries/plain/es/es_ANY.aff /root/hunspell/es_ANY.aff

RUN py.test --pep8
