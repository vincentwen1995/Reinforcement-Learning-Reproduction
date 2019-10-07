from gym import spaces


class BaseMethod(object):

    def __init__(self, args):
        self.args = args

    def select_action(self):
        raise NotImplementedError('Needs to be overridden in subclasses.')

    def update(self):
        raise NotImplementedError('Needs to be overridden in subclasses.')

    def train(self):
        raise NotImplementedError('Needs to be overridden in subclasses.')


def get_space_shape(space):
    if type(space) is spaces.Box:
        return space.shape[0]
    elif type(space) is spaces.Discrete:
        return space.n
