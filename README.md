# Magic

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/pasmod/magic/blob/master/License.md)

Author profiling deals with the study of various profile dimensions of an author such as age and gender. This is the implementation of our methodology proposed for the task of cross-genre author profiling at PAN 2016. We address gender and age prediction as a classification task and approach this problem by extracting stylistic and lexical features for training a logistic regression model.

# Highlights
*  First place for gender detection in English at PAN 2016
*  Second place in terms of joint accuracy in English at PAN 2016
*  First place in Spanish at PAN 2016

## Dependencies
* [docker](https://www.docker.com/)

## How to setup the project:
``` bash
  cd magic
  make build
```
## Corpora
It it assumed that two corpora (PAN14 and PAN16) are placed in the corpora folder. See [here](http://pan.webis.de/) to get the corpora. The following folder structure is expected:
```
magic
│   README.md
│   ...   
│
└───magic
    ├───corpora
        ├───cpan14-author-profiling-training-corpus-2014-04-16
            ├───pan14-author-profiling-training-corpus-english-blogs-2014-04-16
            ├───pan14-author-profiling-training-corpus-english-reviews-2014-04-16
            ├───pan14-author-profiling-training-corpus-english-socialmedia-2014-04-16
            ├───pan14-author-profiling-training-corpus-english-twitter-2014-04-16
            ├───pan14-author-profiling-training-corpus-spanish-blogs-2014-04-16
            ├───pan14-author-profiling-training-corpus-spanish-socialmedia-2014-04-16
            ├───pan14-author-profiling-training-corpus-spanish-twitter-2014-04-16
        ├───pan16-author-profiling-training-corpus-2016-02-29
            ├───pan16-author-profiling-training-corpus-dutch-2016-02-29
            ├───pan16-author-profiling-training-corpus-english-2016-02-29
            ├───pan16-author-profiling-training-corpus-spanish-2016-02-29
```
If you want to use another folder structure, you have to modify the file pan.py in the parsers module.
## Evaluation
``` bash
make run
```
After running the container use the following command to perform evaluations:
``` bash
python evaluate.py \
--train_corpus=pan2016/english/twitter/gender \
--test_corpus=pan2014/english/blogs/gender \
--profiler=english-gender-profiler
```
Take a look at evaluate.py. Using different annotations you can run various evaluation settings.
Runnig the above command will perform a 10-fold cross validation on the training set and evaluate the profiler on the test set.

# Citation
I you want to cite us in your work, please use the following bibtex entry:
``` bash
@INPROCEEDINGS{modbeckcon:2016,
        AUTHOR             = {Pashutan Modaresi and Matthias Liebeck and Stefan Conrad},
        BOOKTITLE          = {Working Notes Papers of the CLEF 2016 Evaluation Labs},
        ISSN               = {1613-0073},
        MONTH              = sep,
        PUBLISHER          = {CLEF and CEUR-WS.org},
        SERIES             = {CEUR Workshop Proceedings},
        TITLE              = {{Exploring the Effects of Cross-Genre Machine Learning for Author Profiling in PAN 2016}},
        YEAR               = {2016}
}
```
