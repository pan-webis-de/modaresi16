#!/usr/bin/env python
import logging
import argparse
import profiler16_un.datasets.pan2014
import profiler16_un.datasets.pan2016
from profiler16_un.profilers.pos_ngram_profiler import POSNGramProfiler
from profiler16_un.profilers.word_slice_profiler import WordSliceProfiler
from profiler16_un.profilers.pos_tag_profiler import POSTagProfiler
from profiler16_un.profilers.spelling_error_profiler import SpellingErrorProfiler
from profiler16_un.profilers.en_gender_profiler import EnglishGenderProfiler
from profiler16_un.benchmarks.sklearn_benchmark import SklearnBenchmark
from profiler16_un.configuration import Configuration


def configure(conf):
    @conf.profiler('last_character_profiler', slice_length=3, slizer='last_chars')
    def build_last_character_profiler(**args):
        return WordSliceProfiler(**args)

    @conf.profiler('first_character_profiler', slice_length=3, slizer='first_chars')
    def build_last_character_profiler(**args):
        return WordSliceProfiler(**args)

    @conf.profiler('pos_tag_profiler_en', language='en')
    def build_pos_tag_profiler(**args):
        return POSTagProfiler(**args)

    @conf.profiler('spelling_error_profiler_en', language='en')
    def build_spelling_error_profiler(**args):
        return SpellingErrorProfiler(**args)

    @conf.profiler('pos_ngram_profiler_en', lang='en', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('pos_ngram_profiler_es', lang='es', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('pos_ngram_profiler_nl', lang='nl', min_n=3, max_n=3, method='logistic_regression')
    def build_pos_ngram_profiler(**args):
        return POSNGramProfiler(**args)

    @conf.profiler('en_gender_profiler', lang='en', min_n=1, max_n=1, method='logistic_regression')
    def build_en_gender_profiler(**args):
        return EnglishGenderProfiler(**args)

    @conf.dataset('pan2014/gender/english/blog', label='gender', types=['blog'], language='english')
    @conf.dataset('pan2014/age/english/blog', label='age_group', types=['blog'], language='english')
    @conf.dataset('pan2014/gender/english/socialmedia', label='gender', types=['socialmedia'], language='english')
    @conf.dataset('pan2014/age/english/socialmedia', label='age_group', types=['socialmedia'], language='english')
    @conf.dataset('pan2014/gender/english/review', label='gender', types=['review'], language='english')
    @conf.dataset('pan2014/age/english/review', label='age_group', types=['review'], language='english')
    def build_dataset_pan14(label=None, types=None, language=None):
        dataset_iterator = profiler16_un.datasets.pan2014.load(label=label, types=types, language=language)
        pred_profile = lambda profiler, X: profiler.predict(X)
        true_profile = lambda Y: Y
        return dataset_iterator, pred_profile, true_profile

    @conf.dataset('pan2016/gender/english/twitter', label='gender', types=['twitter'], language='english')
    @conf.dataset('pan2016/gender/spanish/twitter', label='gender', types=['twitter'], language='spanish')
    @conf.dataset('pan2016/gender/dutch/twitter', label='gender', types=['twitter'], language='dutch')
    @conf.dataset('pan2016/age/english/twitter', label='age_group', types=['twitter'], language='english')
    @conf.dataset('pan2016/age/spanish/twitter', label='age_group', types=['twitter'], language='spanish')
    def build_dataset_pan16(label=None, types=None, language=None):
        dataset_iterator = profiler16_un.datasets.pan2016.load(label=label, types=types, language=language)
        pred_profile = lambda profiler, X: profiler.predict(X)
        true_profile = lambda Y: Y
        return dataset_iterator, pred_profile, true_profile


def pretty_list(items):
    return ', '.join([x for x in items])


if __name__ == '__main__':
    conf = Configuration()
    argparser = argparse.ArgumentParser(description='Author profiling evaluation')
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
    training_dataset_iterator, train_pred_profile, train_true_profile = conf.get_dataset(args.training_corpus)
    if args.test_corpus:
        test_dataset_iterator, test_pred_profile, test_true_profile = conf.get_dataset(args.test_corpus)
    else:
        test_dataset_iterator = None
    profiler_instance = conf.get_profiler(args.profiler_name)
    benchmark = SklearnBenchmark()
    benchmark.run(training_dataset_iterator=training_dataset_iterator,
                  test_dataset_iterator=test_dataset_iterator,
                  profiler=profiler_instance)
