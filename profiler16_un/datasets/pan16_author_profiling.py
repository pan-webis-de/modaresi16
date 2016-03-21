from __future__ import print_function
import os
import errno
import time
from shutil import copyfile
import zipfile
import pandas as pd
from bs4 import BeautifulSoup
import gzip
import cPickle as pickle


VERSION = '0.0.1'
GROUP = 'PAN'
if not os.environ.get('DATASETS', None):
    raise ValueError(
        'You need to set an environment variable DATASETS_REMOTE, containing the path to corpora folder.')
DATASETS_REMOTE = os.environ['DATASETS']
DATASETS_LOCAL = os.path.expanduser('~/.datasets')

AVAILABLE_SECTIONS = {
    'english': ['twitter'],
    'spanish': ['twitter'],
    'dutch': ['twitter']
}


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def fetch_data(remote_path=DATASETS_REMOTE):
    file_name = 'pan16-author-profiling-training-corpus-2016-02-29.zip'
    dst_directory = os.path.join('/tmp', GROUP + '_' + str(time.time()))
    os.makedirs(dst_directory)
    corpus_zip = os.path.join(dst_directory, file_name)
    print(u'Downloading corpus {}'.format(corpus_zip))
    copyfile(os.path.join(remote_path, GROUP, file_name), corpus_zip)
    with open(corpus_zip, 'rb') as file:
        print(u'Unpacking {}'.format(corpus_zip))
        z = zipfile.ZipFile(file)
        for name in z.namelist():
            z.extract(name, dst_directory)
    extraction_folder = os.path.join(dst_directory, 'pan16-author-profiling-training-corpus-2016-02-29')
    for f in os.listdir(extraction_folder):
        subcorpus_zip = os.path.join(extraction_folder, f)
        with open(subcorpus_zip, 'rb') as file:
            print(u'Unpacking {}'.format(subcorpus_zip))
            z = zipfile.ZipFile(file)
            for name in z.namelist():
                z.extract(name, extraction_folder)
    return extraction_folder


def parse_xml(filename):
    """parses xml files, that contains the author and its documents"""
    docs = []
    with open(filename) as fp:
        xml = '\n'.join(fp.readlines())
        soup = BeautifulSoup(xml, 'html.parser')
        author = soup.author
        author_attrs = dict(author.attrs)
        del author_attrs['age_group']
        del author_attrs['gender']
        for d in author.documents.find_all('document'):
            text1 = unicode(d.get_text()).strip()
            text2 = unicode(BeautifulSoup(text1, 'html.parser').get_text())
            text = text2.strip()
            doc = dict(text=text)
            doc.update(d.attrs)
            docs.append(doc)
    return author_attrs, docs


def iterdocs(language,
             sections,
             basedir,
             data_path='',
             subfolder_template='pan16-author-profiling-training-corpus-{language}-2016-02-29'):
    """Yields all documents of a given language"""
    data_path = os.path.join(basedir, data_path)
    counter = 0
    authors_with_missing_text = 0
    for section in sections:
        section_counter = 0
        directory = os.path.join(data_path, subfolder_template.format(language=language, section=section))
        print('Loading (language={})'.format(language, section))
        df = pd.read_csv(os.path.join(directory, 'truth.txt'),
                         sep=':::',
                         engine='python',
                         names=['author_id', 'gender', 'age_group'])
        print('Authors count: {}'.format(len(df)))
        for i in range(len(df)):
            row = df.iloc[i]
            labels = dict(author_id=row['author_id'], gender=row['gender'], age_group=row['age_group'])
            author_attrs, docs = parse_xml(os.path.join(directory, str(labels['author_id']) + '.xml'))
            text = '\n'.join([doc['text'] for doc in docs]).strip()
            if not text:
                authors_with_missing_text += 1
                print('The author with id={} has no text!'.format(labels['author_id']))
            doc = dict(text=text)
            doc.update({'author.' + k: v for k, v in author_attrs.iteritems()})
            doc.update({'label.' + k: v for k, v in labels.iteritems()})
            yield doc
            counter += 1
            section_counter += 1
        print('Authors in section {}: {}'.format(section, section_counter))
    print('Total authors: {}'.format(counter))
    print('Authors with missing text: {}'.format(authors_with_missing_text))


def read_documents(language, sections, basedir):
    docs = [doc for doc in iterdocs(language, sections, basedir)]
    return docs


def convert():
    print('Start corpus conversion')
    extraction_folder = fetch_data()
    for language in AVAILABLE_SECTIONS.keys():
        docs = read_documents(language, AVAILABLE_SECTIONS[language], extraction_folder)
        save(docs, language)


def save(docs, language, version=VERSION):
    mkdir_p(os.path.join(DATASETS_LOCAL, GROUP))
    name = 'prdatasets_pan16_author_profiling-{}-{}.pgz'.format(language, version)
    local_fullname = os.path.join(DATASETS_LOCAL, GROUP, name)
    print('Saving {}'.format(local_fullname))
    with gzip.GzipFile(local_fullname, 'w') as f:
        pickle.dump(docs, f)


def load(language, version=VERSION):
    name = 'datasets_pan16_author_profiling-{}-{}.pgz'.format(language, version)
    local_fullname = os.path.join(DATASETS_LOCAL, GROUP, name)
    if not os.path.isfile(local_fullname):
        remote_fullname = os.path.join(DATASETS_REMOTE, GROUP, name)
        print('Copy {} --> {}'.format(remote_fullname, local_fullname))
        copyfile(remote_fullname, local_fullname)
    print('Loading {}'.format(local_fullname))
    with gzip.GzipFile(local_fullname, 'r') as f:
        return pickle.load(f)


if __name__ == '__main__':
    convert()
    assert 436 == len(load('english'))
    assert 250 == len(load('spanish'))
    assert 384 == len(load('dutch'))
