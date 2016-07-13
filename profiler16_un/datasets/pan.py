from pan_utils import load_xml_dataset
import os


base_2014_path = 'profiler16_un/corpora/pan14-author-profiling-training-corpus-2014-04-16'
en14_blogs_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-english-blogs-2014-04-16')
en14_reviews_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-english-reviews-2014-04-16')
en14_socialmedia_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-english-socialmedia-2014-04-16')
en14_twitter_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-english-twitter-2014-04-16')
es14_blogs_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-spanish-blogs-2014-04-16')
es14_socialmedia_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-spanish-socialmedia-2014-04-16')
es14_twitter_path = os.path.join(base_2014_path, 'pan14-author-profiling-training-corpus-spanish-twitter-2014-04-16')

base_2016_path = 'profiler16_un/corpora/pan16-author-profiling-training-corpus-2016-02-29'
en16_twitter_path = os.path.join(base_2016_path, 'pan16-author-profiling-training-corpus-english-2016-02-29')
es16_twitter_path = os.path.join(base_2016_path, 'pan16-author-profiling-training-corpus-spanish-2016-02-29')
nl16_twitter_path = os.path.join(base_2016_path, 'pan16-author-profiling-training-corpus-spanish-2016-02-29')


mapping = {'english14': {'socialmedia': en14_socialmedia_path,
                         'twitter': en14_twitter_path,
                         'blogs': en14_blogs_path,
                         'reviews': en14_reviews_path},
           'spanish14': {'blogs': es14_blogs_path,
                         'socialmedia': es14_socialmedia_path,
                         'twitter': es14_twitter_path},
           'english16': {'twitter': en16_twitter_path},
           'spanish16': {'twitter': es16_twitter_path},
           'spanish16': {'twitter': nl16_twitter_path}
           }


def load(label='gender', type='socialmedia', language='english', year='2016'):
    X, y = load_xml_dataset(mapping[language + year[2:4]][type])
    return X, y
