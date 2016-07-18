#!/usr/bin/env python
import logging
import argparse
from profiler16_un.profilers.pos_ngram_profiler import POSNGramProfiler
from profiler16_un.profilers.embeddings_profiler import EmbeddingsProfiler
from profiler16_un.profilers.pos_tag_profiler import POSTagProfiler
from profiler16_un.profilers.spelling_error_profiler import SpellingErrorProfiler
from profiler16_un.profilers.en_gender_profiler import EnglishGenderProfiler
from profiler16_un.benchmarks.sklearn_benchmark import SklearnBenchmark
from profiler16_un.datasets.pan import load
from profiler16_un.configuration import Configuration


def configure(conf):
    @conf.profiler('pos_tag_profiler_en', language='en')
    def build_pos_tag_profiler(**args):
        return POSTagProfiler(**args)

    @conf.profiler('spelling_error_profiler_en', language='en')
    def build_spelling_error_profiler(**args):
        return SpellingErrorProfiler(**args)

    @conf.profiler('embeddings_profiler_en', language='en')
    def build_embeddings_profiler(**args):
        return EmbeddingsProfiler(**args)

    @conf.profiler('pos_ngram_profiler_en', lang='en', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('pos_ngram_profiler_es', lang='es', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('pos_ngram_profiler_nl', lang='nl', min_n=3, max_n=3, method='logistic_regression')
    def build_pos_ngram_profiler(**args):
        return POSNGramProfiler(**args)

    @conf.profiler('en_gender_profiler', lang='en', method='logistic_regression')
    def build_en_gender_profiler(**args):
        fns_gender = ['unigram', 'bigram', 'spelling', 'char']
        fns_age = ['unigram', 'bigram', 'spelling', 'punctuation', 'char']
        return EnglishGenderProfiler(lang='en', method='logistic_regression', feature_names=fns_gender)

    @conf.dataset('pan2014/english/blogs/gender', label='gender', type='blogs', language='english', year='2014')
    @conf.dataset('pan2014/english/blogs/age', label='age_group', type='blogs', language='english', year='2014')
    @conf.dataset('pan2014/english/socialmedia/gender', label='gender', type='socialmedia', language='english', year='2014')
    @conf.dataset('pan2014/english/socialmedia/age', label='age_group', type='socialmedia', language='english', year='2014')
    @conf.dataset('pan2014/english/review/gender', label='gender', type='review', language='english', year='2014')
    @conf.dataset('pan2014/english/review/age', label='age_group', type='review', language='english', year='2014')
    @conf.dataset('pan2014/spanish/blogs/gender', label='gender', type='blogs', language='spanish', year='2014')
    @conf.dataset('pan2014/spanish/blogs/age', label='age_group', type='blogs', language='spanish', year='2014')
    @conf.dataset('pan2014/spanish/socialmedia/gender', label='gender', type='socialmedia', language='spanish', year='2014')
    @conf.dataset('pan2014/spanish/socialmedia/age', label='age_group', type='socialmedia', language='spanish', year='2014')
    def build_dataset_pan14(label=None, type=None, language=None, year=None):
        X, y = load(label=label, type=type, language=language)
        X = [x['text'] for x in X]
        y = [yy[label]for yy in y]
        return X, y

    @conf.dataset('pan2016/english/twitter/gender', label='gender', type='twitter', language='english', year='2016')
    @conf.dataset('pan2016/spanish/twitter/gender', label='gender', type='twitter', language='spanish', year='2016')
    @conf.dataset('pan2016/dutch/twitter/gender', label='gender', type='twitter', language='dutch', year='2016')
    @conf.dataset('pan2016/english/twitter/age', label='age_group', type='twitter', language='english', year='2016')
    @conf.dataset('pan2016/spanish/twitter/age', label='age_group', type='twitter', language='spanish', year='2016')
    def build_dataset_pan16(label=None, type=None, language=None, year=None):
        X, y = load(label=label, type=type, language=language)
        X = [x['text'] for x in X]
        y = [yy[label]for yy in y]
        return X, y


def pretty_list(items):
    return ', '.join([x for x in items])


if __name__ == '__main__':
    conf = Configuration()
    argparser = argparse.ArgumentParser(
        description='Author profiling evaluation')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')

    argparser.add_argument('-c', '--train_corpus', dest='training_corpus', type=str, required=True,
                           help='Set name of the training corpus used for the evaluation: ' + pretty_list(
                               conf.get_dataset_names()))

    argparser.add_argument('-t', '--test_corpus', dest='test_corpus', type=str, required=False,
                           help='Set name of the test corpus used for the evaluation: ' + pretty_list(
                               conf.get_dataset_names()))

    argparser.add_argument('-p', '--profiler', dest='profiler_name', type=str, required=True,
                           help='Name of the invoked profiler: ' + pretty_list(conf.get_profiler_names()))

    args = argparser.parse_args()
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)

    configure(conf)
    X_train, y_train = conf.get_dataset(
        args.training_corpus)
    if args.test_corpus:
        X_test, y_test = conf.get_dataset(args.test_corpus)
    else:
        X_test, y_test = None
    profiler_instance = conf.get_profiler(args.profiler_name)
    benchmark = SklearnBenchmark()
    benchmark.run(X_train=X_train, y_train=y_train,
                  X_test=X_test, y_test=y_test,
                  profiler=profiler_instance)
