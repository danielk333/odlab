class StateGenerator:
    def get_states(self, state0, times):
        raise NotImplementedError()


class sortsPropagator(StateGenerator):
    def __init__(self, epoch, propagator, propagator_args={}):
        self.propagator = propagator
        self.epoch = epoch

        self.propagator_args = propagator_args

    def get_states(self, state0, times):
        times = times
        t = (times - self.epoch).sec
        return self.propagator.propagate(t, state0, self.epoch, **self.propagator_args)
