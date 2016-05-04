import os
from bs4 import BeautifulSoup
import warnings
import logging

warnings.filterwarnings('error')

logger = logging.getLogger(__name__)

TRUTH_FILE_COLUMNS = ['id', 'gender', 'age_group',
                      'extroverted', 'stable', 'agreeable', 'conscientious', 'open']


def parse_xml(filename, parser='html.parser', clean_text=True):
    """parses xml files, that contains the author and its documents"""
    base = os.path.basename(filename)
    idx = str(base[:-4])
    posts = []
    logger.debug(u'Parse {}'.format(filename))
    with open(filename) as fp:
        xml = '\n'.join(fp.readlines())
        soup = BeautifulSoup(xml, parser)
        author = soup.author
        author_attrs = dict(author.attrs)
        for d in author.find_all('document'):
            text = unicode(d.get_text()).strip()
            if clean_text:
                try:
                    post_soup = BeautifulSoup(text, 'html.parser')
                    text = unicode(post_soup.get_text()).strip()
                except UserWarning:
                    pass
            if not text:
                logger.debug(u'No text in document: {}'.format(unicode(d)))
            post = dict(text=text)
            post.update(d.attrs)
            post['markup'] = str(d)
            posts.append(post)
    text = '\n'.join([post['text'] for post in posts]).strip()
    doc = dict()
    doc['id'] = idx
    doc['posts'] = posts
    doc['text'] = text
    doc.update({'attr.' + k: v for k, v in author_attrs.iteritems()})

    return doc


def parse_xml_files(xml_dir):
    import multiprocessing
    logger.info('Reading xml files from: {}'.format(xml_dir))

    filenames = sorted([os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')])
    logger.info('Files: {}'.format(len(filenames)))

    cpu_count = multiprocessing.cpu_count()
    n_jobs = cpu_count - 1 if cpu_count >= 2 else 1
    logger.info('Parallel jobs: {}'.format(n_jobs))

    docs = []
    total = len(filenames)
    if n_jobs == 1:
        for xml_filename in filenames:
            doc = parse_xml(xml_filename)
            docs.append(doc)
    else:
        pool = multiprocessing.Pool(n_jobs)
        docs = pool.map(parse_xml, filenames)
    logger.info('Parsed {} xml files in {}'.format(len(docs), xml_dir))
    return docs


def read_truth_file(truth_file, column_names=TRUTH_FILE_COLUMNS):
    logger.info('Parsing truth file: {}'.format(truth_file))

    doc_labels = []
    with open(truth_file) as fp:
        for line in fp.readlines():
            row = line.split(':::')
            if u'M' == row[1]:
                row[1] = u'MALE'
            if u'F' == row[1]:
                row[1] = u'FEMALE'
            labels = {}
            for i, v in enumerate(row):
                k = column_names[i]
                labels[k] = str(v).strip()
            doc_labels.append(labels)
    return doc_labels


def detect_language(xml_dir):
    base = str(os.path.basename(xml_dir))
    languages = ['english', 'spanish', 'dutch', 'italian']
    langcodes = ['en', 'es', 'nl', 'it']
    for language, code in zip(languages, langcodes):
        if language in base:
            logger.info('Detected language: {}'.format(code))
            return code
    raise ValueError('Could not detect language in xml base folder: {}'.format(base))


def detect_type(xml_dir):
    base = str(os.path.basename(xml_dir))
    sections = ['blogs', 'reviews', 'socialmedia', 'twitter']
    types = ['blog', 'review', 'socialmedia', 'twitter']
    fallback = 'twitter'
    for s, t in zip(sections, types):
        if s in base:
            logger.info('Detected type: {}'.format(t))
            return t
    logger.info('Fallback to type: {}'.format(fallback))
    return fallback


def concat_texts(posts):
    return u'\n'.join([unicode(post['text']) for post in posts]).strip()


def load_xml_dataset(xml_dir):
    X = parse_xml_files(xml_dir)
    if X[0].get('attr.type', None) is None:
        atype = detect_type(xml_dir)
        logger.info('Adding missing author type information to all samples: {}'.format(atype))
        for doc in X:
            doc['attr.type'] = atype
    if X[0].get('attr.lang', None) is None:
        lang = detect_language(xml_dir)
        logger.info('Adding missing author language information to all samples: {}'.format(lang))
        for doc in X:
            doc['attr.lang'] = lang
    for doc in X:
        text_len = len(concat_texts(doc['posts']))
        logger.info(u'Instance id={idx}, type={atype}, lang={lang}, chars={chars}'.format(idx=doc['id'], atype=doc['attr.type'], lang=doc['attr.lang'], chars=text_len))
    y = None
    truth_file = os.path.join(xml_dir, 'truth.txt')
    if os.path.isfile(truth_file):
        logger.info('Detected truth file: {}'.format(truth_file))
        y = read_truth_file(truth_file)
        logger.info(y)
        # sort y with respect to X
        y_sort_index = {doc['id']: i for i, doc in enumerate(X)}
        y.sort(key=lambda labels: y_sort_index[labels['id']])
        # Alternatively, we can sort X with respect to y
        # y_sort_index = {labels['id']: i for i, labels in enumerate(y)}
        # X.sort(key=lambda doc: y_sort_index[doc['id']])
    return X, y


def save_output_xmls(xml_dir, X, y_pred):
    if not xml_dir:
        raise ValueError('Specify the xml directory')
    for author, labels in zip(X, y_pred):
        soup = BeautifulSoup(features='xml')
        tag = soup.new_tag('author',
                           id=author['id'],
                           type=author['attr.type'],
                           lang=author['attr.lang'].lower(),
                           age_group=labels['age_group'].lower(),
                           gender=labels['gender'].lower())
        soup.append(tag)
        output_xml_filename = os.path.join(xml_dir, str(author['id']) + '.xml')
        logger.info('Write {} <-- {}'.format(output_xml_filename, str(tag)))
        with open(output_xml_filename, 'w') as fp:
            markup = soup.prettify()
            fp.write(markup)
    logger.info('Wrote {} files'.format(len(X)))
