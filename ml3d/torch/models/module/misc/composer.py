import collections

from ..builder import COMPOSER


class Composer(object):
    def __init__(self, modules):
        assert isinstance(modules, collections.abc.Sequence)
        self.modules = []
        for m in modules:
            if callable(m):
                self.modules.append(m)
            else:
                raise TypeError('modules must be callable')

    def __call__(self, data):
        for m in self.modules:
            data = m(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

class aaa(object):
    """docstring for aaa"""
    def __init__(self, arg):
        super(aaa, self).__init__()
        self.arg = arg
        
