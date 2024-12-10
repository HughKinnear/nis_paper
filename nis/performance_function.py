from copy import deepcopy

class PerformanceFunction:

    def __init__(self, performance_function):
        self.non_cache_performance_function = deepcopy(performance_function)
        self.cache  = {}
        self.names = []
        self.saved_eval_counts = []

    def __call__(self, x):
        tuple_x = tuple(x)
        try:
            result = self.cache[tuple_x]
        except KeyError:
            self.cache[tuple_x] = result = self.non_cache_performance_function(tuple_x)
        return result

    @property
    def eval_count(self):
        return len(self.cache)
    
    def save(self,name):
        self.names.append(name)
        self.saved_eval_counts.append(self.eval_count
                                      - sum(self.saved_eval_counts))
        
    @property
    def eval_count_dict(self):
        result = {}
        for string, number in zip(self.names, self.saved_eval_counts):
            if string in result:
                result[string] += number
            else:
                result[string] = number
        result['Total'] = self.eval_count
        return result