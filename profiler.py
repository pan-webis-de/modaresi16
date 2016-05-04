#!/usr/bin/env python
import argparse
import logging
from profiler16_un.datasets.pan_utils import load_xml_dataset, save_output_xmls
from profiler16_un.profilers.en_gender_profiler import EnglishGenderProfiler

base = '/media/training-datasets/author-profiling'
en_corpus = '/media/en'
es_corpus = base + '/pan16-author-profiling-training-dataset-english-2016-04-25'
nl_corpus = base + '/pain16-author-profiling-training-dataset-english-2016-04-25'


def main(tira_input=None):
    X_test, Y_test = load_xml_dataset(tira_input)
    X_test_txt = [x['text'] for x in X_test]
    lang = X_test[0]['attr.lang'].lower()

    if 'en' == lang:
        X_train, Y_train = load_xml_dataset(en_corpus)

    if 'es' == lang:
        X_train, Y_train = load_xml_dataset(es_corpus)

    if 'nl' == lang:
        X_train, Y_train = load_xml_dataset(nl_corpus)

    X_train_txt = [x['text'] for x in X_train]
    Y_train_gender = [y['gender'] for y in Y_train]
    Y_train_age = [y['age_group'] for y in Y_train]

    p = EnglishGenderProfiler(method='logistic_regression')
    p.train(X_train_txt, Y_train_gender)
    Y_pred_gender = p.predict(X_test_txt)

    p = EnglishGenderProfiler(method='logistic_regression')
    p.train(X_train_txt, Y_train_age)
    Y_pred_age = p.predict(X_test_txt)

    Y_pred = [{'gender': y_pred_gender, 'age_group': y_pred_age} for (y_pred_gender, y_pred_age) in zip(Y_pred_gender, Y_pred_age)]
    save_output_xmls(args.tira_output, X_test, Y_pred)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Author Profiler for PAN 2016')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')

    argparser.add_argument('-i', '--tira_input', dest='tira_input', type=str, required=True,
                           help='Path to the corpus for which the gender and age of the authors have to be predicted')

    argparser.add_argument('-o', '--tira_output', dest='tira_output', type=str, required=True,
                           help='Output directory')

    args = argparser.parse_args()
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)
    main(args.tira_input)
