#!/usr/bin/env python
import argparse
import logging
from profiler16_un.datasets.pan_utils import load_xml_dataset, save_output_xmls
from profiler16_un.profilers.en_gender_profiler import EnglishGenderProfiler


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Author Profiler for PAN 2016')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')

    argparser.add_argument('-i', '--tira_input', dest='tira_input', type=str, required=True,
                           help='Path to the corpus for which the gender and age of the authors have to be predicted')

    argparser.add_argument('-o', '--tira_output', dest='tira_output', type=str, required=True,
                           help='Output directory')

    p = EnglishGenderProfiler(method='logistic_regression')
    args = argparser.parse_args()
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)
    X, y = load_xml_dataset(args.tira_input)
    texts = [x['text'] for x in X]
    y = [yy['gender'] for yy in y]
    p.train(texts, y)
    y_pred_gender = p.predict(texts)
    y_pred = [{'gender': y, 'age_group': '20-30'} for y in y_pred_gender]
    save_output_xmls(args.tira_output, X, y_pred)
