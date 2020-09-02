#coding: future_fstrings
import os.path
import shutil
import sys
import tempfile
import yaml
from importlib import import_module
from addict import Dict


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


class Config(object):
    """docstring for Config"""
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict shoud be a dict, but'
                            f'got {type(cfg_dict)}')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))

        self.cfg_dict = cfg_dict

    def dump(self, *args, **kwargs):
        """Dump to a string."""
        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, ConfigDict):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self._cfg_dict, [])
        print(self_as_dict)
        return yaml.dump(self_as_dict, *args, **kwargs)
        #return self_as_dict


    def merge_from_dict(self, new_dict):
        """Merge a new into cfg_dict.

        Args:
            new_dict (dict): a dict of configs.
        """
        b = self.copy()
        for k, v in new_dict.items():
            if v is None:
                continue
            b[k] = v
        return Config(b)


    @staticmethod
    def merge_default_cfgs(
            default_cfg_path, 
            cfg, 
            **kwargs):
        result_cfg = Config.load_from_file(default_cfg_path)
  
        if cfg is not None:
            if isinstance(cfg, str):
                result_cfg = Config.load_from_file(cfg)
            elif isinstance(cfg, Config):
                result_cfg = cfg
            elif isinstance(cfg, dict):
                result_cfg = result_cfg.merge_from_dict(cfg)
            else:
                raise TypeError("cfg must be a string, dict, or Config " +
                                "but got {}".format(type(cfg)))

        result_cfg.merge_from_dict(kwargs)

        return result_cfg


    @staticmethod
    def load_from_file(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} not found')

        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py')
                temp_config_name = os.path.basename(temp_config_file.name)
                shutil.copyfile(
                    filename, os.path.join(temp_config_dir, temp_config_name))
                temp_module_name = os.path.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
                # close temp file
                temp_config_file.close()

        if filename.endswith('.yaml') or filename.endswith('.yml') :
            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)

        return Config(cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)
