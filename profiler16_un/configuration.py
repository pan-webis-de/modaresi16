import logging

logger = logging.getLogger(__name__)


class Configuration():
    def __init__(self):
        self.profiler_registry = {}
        self.dataset_registry = {}

    def profiler(self, name, **args):
        logger.debug('Register profiler {}, opt={}'.format(name, args))

        def decorator(f):
            if name in self.profiler_registry.keys():
                raise ValueError('The profiler {} is already registered. Please use another name!'.format(name))

            def wrapper():
                return f(**args)
            self.profiler_registry[name] = wrapper
            return f
        return decorator

    def get_profiler(self, name):
        builder = self.profiler_registry.get(name)
        if builder:
            return builder()
        else:
            raise ValueError("Profiler not found: {}".format(name))

    def get_profiler_names(self):
        return self.profiler_registry.keys()

    def dataset(self, name, **args):
        logger.debug('Register dataset {}, opt={}'.format(name, args))

        def decorator(f):
            if name in self.dataset_registry.keys():
                raise ValueError('The dataset {} is already registered. Please use another name!'.format(name))

            def wrapper():
                return f(**args)
            self.dataset_registry[name] = wrapper
            return f
        return decorator

    def get_dataset(self, name):
        builder = self.dataset_registry.get(name)
        if builder:
            return builder()
        else:
            raise ValueError("Dataset not found: {}".format(name))

    def get_dataset_names(self):
        return self.dataset_registry.keys()
