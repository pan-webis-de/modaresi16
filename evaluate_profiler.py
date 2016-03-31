#!/usr/bin/env python
import logging
import argparse
import profiler16_un.datasets.pan2014
import profiler16_un.datasets.pan2016
from profiler16_un.profilers.random_profiler import RandomProfiler
from profiler16_un.profilers.character_ngram_profiler import CharacterNGramProfiler
from profiler16_un.profilers.pos_ngram_profiler import POSNGramProfiler
from profiler16_un.profilers.logistic_regression_profiler import LogisticRegressionProfiler
from profiler16_un.profilers.random_forest_profiler import RandomForestProfiler
from profiler16_un.profilers.aleksey_profiler import AlekseyProfiler
from profiler16_un.profilers.word_slice_profiler import WordSliceProfiler
from profiler16_un.profilers.pos_tag_profiler import POSTagProfiler
from profiler16_un.playbooks.accumulate_benchmark import AccumulateBenchmark
from profiler16_un.playbooks.sklearn_benchmark import SklearnBenchmark
from profiler16_un.configuration import Configuration
from profiler16_un.metrics.zero_one import ZeroOne


def configure(conf):

    @conf.profiler('character_ngram_3_3_logistic_regression', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('character_ngram_3_3_random_forest', min_n=3, max_n=3, method='random_forest')
    def build_character_ngram_profiler(**args):
        return CharacterNGramProfiler(**args)

    @conf.profiler('random')
    def build_random_profiler(**args):
        return RandomProfiler(**args)

    @conf.profiler('logistic_regression')
    def build_logistic_regression_profiler(**args):
        return LogisticRegressionProfiler(**args)

    @conf.profiler('random_forest')
    def build_random_forest_profiler(**args):
        return RandomForestProfiler(**args)

    @conf.profiler('aleksey_profiler')
    def build_aleksey_proilfer(**args):
        return AlekseyProfiler(**args)

    @conf.profiler('last_character_profiler', slice_length=3, slizer='last_chars')
    def build_last_character_profiler(**args):
        return WordSliceProfiler(**args)

    @conf.profiler('first_character_profiler', slice_length=3, slizer='first_chars')
    def build_last_character_profiler(**args):
        return WordSliceProfiler(**args)

    @conf.profiler('pos_tag_profiler_en', absolute_frequency=True, normalize=False, language='en')
    def build_pos_tag_profiler(**args):
        return POSTagProfiler(**args)

    @conf.profiler('pos_ngram_profiler_en', lang='en', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('pos_ngram_profiler_es', lang='es', min_n=3, max_n=3, method='logistic_regression')
    @conf.profiler('pos_ngram_profiler_nl', lang='nl', min_n=3, max_n=3, method='logistic_regression')
    def build_pos_ngram_profiler(**args):
        return POSNGramProfiler(**args)

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

    @conf.benchmark('accumulate')
    def build_accumulate_benchmark(**args):
        return AccumulateBenchmark(**args)

    @conf.benchmark('sklearn')
    def build_sklearn_benchmark(**args):
        return SklearnBenchmark(**args)

    @conf.metric('zero_one')
    def build_zero_one_metric(**args):
        return ZeroOne(**args)


def pretty_list(items):
    return ', '.join([x for x in items])


if __name__ == '__main__':
    conf = Configuration()
    argparser = argparse.ArgumentParser(description='Author profiling evaluation')
    argparser.add_argument('-l', '--log-level', dest='log_level', type=str, default='INFO',
                           help='Set log level (DEBUG, INFO, ERROR)')
    argparser.add_argument('-f', '--log-freq', dest='log_freq', type=int, default=1000,
                           help='Set log progress frequency')
    argparser.add_argument('-c', '--corpus', dest='corpus_name', type=str, required=True,
                           help='Set name of the corpus used for the evaluation: ' + pretty_list(conf.get_dataset_names()))
    argparser.add_argument('-s', '--profiler', dest='profiler_name', type=str, required=True,
                           help='Name of the invoked profiler: ' + pretty_list(conf.get_profiler_names()))
    argparser.add_argument('-m', '--metric', dest='metric_name', type=str, required=True,
                           help='Name of the applied metric: ' + pretty_list(conf.get_metric_names()))
    argparser.add_argument('-b', '--benchmark', dest='benchmark_name', type=str, required=True,
                           help='Name of the applied benchmark: ' + pretty_list(conf.get_benchmark_names()))
    args = argparser.parse_args()
    LOGFMT = '%(asctime)s %(name)s %(levelname)s %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOGFMT)

    configure(conf)
    dataset_iterator, pred_profile, true_profile = conf.get_dataset(args.corpus_name)
    profiler_instance = conf.get_profiler(args.profiler_name)
    metric_instance = conf.get_metric(args.metric_name)
    benchmark = conf.get_benchmark(args.benchmark_name)
    benchmark.run(dataset_iterator=dataset_iterator,
                  profiler=profiler_instance,
                  metric=metric_instance,
                  pred_profile=pred_profile,
                  true_profile=true_profile,
                  n_log_freq=args.log_freq)
