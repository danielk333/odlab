import time
from functools import wraps


class BaseProfiler:
    def __init__(self):
        self.executions = {}
        self.times = {}
        self.names = []
        self.enabled = {}
        self.timings = {}

    def start(self, name):
        self.timings[name] = time.time()

    def stop(self, name):
        dt = time.time() - self.timings[name]
        self.timings[name] = None
        self.add(name, dt)
        return dt

    def init(self, name, enabled):
        self.names.append(name)
        self.enabled[name] = enabled
        self.executions[name] = 0
        self.times[name] = 0.0

    def add(self, name, dt):
        if self.enabled[name]:
            self.executions[name] += 1
            self.times[name] += dt

    def __str__(self):
        means = self.means()
        max_len = max([len(x) for x in self.names])
        rows = [
            f'{name:<{max_len}} @ {self.executions[name]} x \
            {means.get(name,0):.3e} s = {self.times[name]:.3e} s total'
            for name in self.names
            if self.executions[name] > 0
        ]
        return '\n'.join(rows)

    def means(self):
        return {
            name: self.times[name]/self.executions[name]
            for name in self.names 
            if self.executions[name] > 0
        }

    def enable(self, prefix):
        for name in self.names:
            if name.startswith(prefix):
                self.enabled[name] = True


PROFILER = BaseProfiler()


def timeing(prefix, enabled=False):
    '''Decorator that adds executing timing
    '''

    def timeing_wrapper(func):
        name = f'{prefix}.{func.__name__}'
        PROFILER.init(name, enabled)

        @wraps(func)
        def timed_func(*args, **kwargs):
            if PROFILER.enabled[name]:
                t0 = time.time()

            ret = func(*args, **kwargs)

            if PROFILER.enabled[name]:
                dt = time.time() - t0
                PROFILER.add(name, dt)

            return ret

        return timed_func
    return timeing_wrapper
