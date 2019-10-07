from gym import spaces


class BaseMethod(object):
    """Base class for different actor-critic methods (maybe only used by GAE).        
    """

    def __init__(self):
        """Contructor with parsed args.
        """

    def select_action(self):
        raise NotImplementedError('Needs to be overridden in subclasses.')

    def update(self):
        raise NotImplementedError('Needs to be overridden in subclasses.')

    def train(self):
        raise NotImplementedError('Needs to be overridden in subclasses.')


def get_space_shape(space):
    """Dynamically get the shape of the space (observation_space/action_space).
    TODO: Needs extending of compatibility for other gym.space objects.

    Args:
        space (gym.space.Space): observation_space/action_space

    Returns:
        [int]: space size
    """
    if type(space) is spaces.Box:
        return space.shape[0]
    elif type(space) is spaces.Discrete:
        return space.n
