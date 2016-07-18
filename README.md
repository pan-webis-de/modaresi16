# Magic

Author Profiler for PAN 2016
  

How to run the experiments:
* make build (only once!)
* make run (to run the container)

Then you can start to run different experiments:

``` bash
python evaluate_profiler.py --train_corpus=pan2016/gender/english/twitter --test_corpus=pan2016/gender/english/twitter  --profiler=en_gender_profiler
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
