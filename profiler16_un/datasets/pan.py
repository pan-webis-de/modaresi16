from pan_utils import load_xml_dataset
import os


base_path = 'profiler16_un/corpora/pan14-author-profiling-training-corpus-2014-04-16'
en_blogs_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-english-blogs-2014-04-16')
en_reviews_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-english-reviews-2014-04-16')
en_socialmedia_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-english-socialmedia-2014-04-16')
en_twitter_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-english-twitter-2014-04-16')
es_blogs_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-spanish-blogs-2014-04-16')
es_socialmedia_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-spanish-socialmedia-2014-04-16')
es_twitter_path = os.path.join(base_path, 'pan14-author-profiling-training-corpus-spanish-twitter-2014-04-16')
mapping = {'english': {'socialmedia': en_socialmedia_path,
                       'twitter':en_twitter_path,
                       'blogs':en_blogs_path,
                       'reviews':en_reviews_path},
           'spanish': {'blogs':es_blogs_path,
                       'socialmedia':es_socialmedia_path,
                       'twitter':es_twitter_path}
           }


def load(label='gender', type='socialmedia', language='english'):
    X, y = load_xml_dataset(mapping[language][type])
