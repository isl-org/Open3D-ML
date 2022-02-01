import inspect


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def get(self, key, framework):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        if framework is None:
            return self._module_dict.get(key, None)
        else:
            if not isinstance(framework, str):
                raise TypeError("framework must be a string, "
                                "either tf or torch, but got {}".format(
                                    type(framework)))
            return self._module_dict[framework].get(key, None)

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class, framework=None, module_name=None):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, "
                            "but got {}".format(type(module_class)))

        if module_name is None:
            module_name = module_class.__name__
        if framework is None:
            self.module_dict[module_name] = module_class
        else:
            if not isinstance(framework, str):
                raise TypeError("framework must be a string, "
                                "either tf or torch, but got {}".format(
                                    type(framework)))
            if framework in self.module_dict:
                self.module_dict[framework][module_name] = module_class
            else:
                self.module_dict[framework] = dict()
                self.module_dict[framework][module_name] = module_class

    def register_module(self, framework=None, name=None):

        def _register(cls):
            self._register_module(cls, framework=framework, module_name=name)

        return _register


def get_from_name(module_name, registry, framework):
    """Build a module from config dict.

    Args:
        module_name (string): Name of the module.
        registry: The registry to search the type from.
        framework (string): Framework, one of 'tf' or 'torch'

    Returns:
        object: The constructed object.
    """
    if not isinstance(module_name, str):
        raise TypeError("module_name must be a string".format(
            type(module_name)))
    if not isinstance(registry, Registry):
        raise TypeError("registry must be an Registry object, "
                        "but got {}".format(type(module_name)))

    obj_cls = registry.get(module_name, framework)
    if obj_cls is None:
        raise KeyError("{} - {} is not in the {} registry".format(
            module_name, framework, registry.name))

    return obj_cls
