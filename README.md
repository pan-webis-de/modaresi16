# Profiler16_un

Author Profiler for PAN 2016
  
How to setup the project:
* Clone the [corpora](https://github.com/pasmod/corpora) repository that contains the data sets
* Define an environment variable that points to the cloned repository. Add the following line to your .bashrc file
  * export DATASETS=[path to the cloned corpora repository]

How to run the experiments:
* make build (only once!)
* make run (to run the container)

Then you can start to run different experiments:

``` bash
python evaluate_profiler.py 
--corpus=pan2014/gender/english/blog
--metric=zero_one
--benchmark=sklearn
--profiler=logistic_regression
```
The result should be something like this:
```
**************************************************
Confusion Matrix
**************************************************
Predicted  FEMALE  MALE  __all__
Actual                          
FEMALE         26    11       37
MALE           11    26       37
__all__        37    37       74
**************************************************
Accuracy: 0.702702702703
**************************************************
```
