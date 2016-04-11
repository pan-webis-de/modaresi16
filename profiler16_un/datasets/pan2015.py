#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pan15_author_profiling


def load(label=None, language=None):
    """Returns the PAN 2015 dataset X, Y. Here X is the list of texts and Y the list of labels.
       The default label is 'gender'. Alternatively, choose 'age_group'.
       The section is restricted to 'blogs', but several are available:
       English: blogs, reviews, socialmedia, twitter
       Spanish: blogs, socialmedia, twitter
    """
    docs = pan15_author_profiling.load(language)
    for doc in docs:
        yield (doc['text'], doc['label.' + label])
