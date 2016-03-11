import random

AGE_GROUPS = ['18-24', '25-34', '35-49', '50-64', '65-xx']
GENDERS = ['MALE', 'FEMALE']


class RandomProfiler():
    def predict(self, X, label='gender'):
        if 'age' == label:
            return random.choice(AGE_GROUPS)
        elif 'gender' == label:
            return random.choice(GENDERS)
