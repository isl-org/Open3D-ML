import os.path
import shutil
import sys
import tempfile
import yaml
from pathlib import Path
from collections import abc
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

    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict should be a dict, but'
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

    def convert_to_tf_names(self, name):
        """Convert keys compatible with tensorflow."""
        cfg = self._cfg_dict
        with open(
                os.path.join(
                    Path(__file__).parent, '../configs/torch_to_tf.yml')) as f:
            mapping = yaml.safe_load(f)[name]

        def convert_dict(cfg, mapping):
            cfg_new = {}
            for key in cfg:
                if isinstance(cfg[key], dict):
                    cfg_new[key] = convert_dict(cfg[key], mapping)
                elif key in mapping:
                    item = cfg[key]
                    if isinstance(mapping[key], list):
                        for k, v in zip(mapping[key], item):
                            cfg_new[k] = v
                    else:
                        cfg_new[mapping[key]] = item
                else:
                    cfg_new[key] = cfg[key]
            return cfg_new

        cfg = convert_dict(cfg, mapping)
        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg))

    @staticmethod
    def merge_cfg_file(cfg, args, extra_dict):
        """Merge args and extra_dict from the input arguments.

        Merge the dict parsed by MultipleKVAction into this cfg.
        """
        # merge args to cfg
        if args.device is not None:
            cfg.pipeline.device = args.device
            cfg.model.device = args.device
        if args.split is not None:
            cfg.pipeline.split = args.split
        if args.main_log_dir is not None:
            cfg.pipeline.main_log_dir = args.main_log_dir
        if args.dataset_path is not None:
            cfg.dataset.dataset_path = args.dataset_path
        if args.ckpt_path is not None:
            cfg.model.ckpt_path = args.ckpt_path

        extra_cfg_dict = {'model': {}, 'dataset': {}, 'pipeline': {}}

        for full_key, v in extra_dict.items():
            d = extra_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict_dataset = Config._merge_a_into_b(extra_cfg_dict['dataset'],
                                                  cfg.dataset)
        cfg_dict_pipeline = Config._merge_a_into_b(extra_cfg_dict['pipeline'],
                                                   cfg.pipeline)
        cfg_dict_model = Config._merge_a_into_b(extra_cfg_dict['model'],
                                                cfg.model)

        return cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model

    @staticmethod
    def merge_module_cfg_file(args, extra_dict):
        """Merge args and extra_dict from the input arguments.

        Merge the dict parsed by MultipleKVAction into this cfg.
        """
        # merge args to cfg
        cfg_dataset = Config.load_from_file(args.cfg_dataset)
        cfg_model = Config.load_from_file(args.cfg_model)
        cfg_pipeline = Config.load_from_file(args.cfg_pipeline)

        cfg_dict = {
            'dataset': cfg_dataset.cfg_dict,
            'model': cfg_model.cfg_dict,
            'pipeline': cfg_pipeline.cfg_dict
        }
        cfg = Config(cfg_dict)

        return Config.merge_cfg_file(cfg, args, extra_dict)

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict `a` into dict `b` (non-inplace). values in `a` will
        # overwrite `b`.
        # copy first to avoid inplace modification
        # from mmcv mmcv/utils/config.py
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict):
                if k in b and not isinstance(b[k], dict):
                    raise TypeError(
                        "{}={} in child config cannot inherit from base ".
                        format(k, v) +
                        "because {} is a dict in the child config but is of ".
                        format(k) +
                        "type {} in base config.  ".format(type(b[k])))
                b[k] = Config._merge_a_into_b(v, b.get(k, ConfigDict()))
            else:
                if v is None:
                    continue
                if v.isnumeric():
                    v = int(v)
                elif v.replace('.', '').isnumeric():
                    v = float(v)
                elif v == 'True' or v == 'true':
                    v = True
                elif v == 'False' or v == 'false':
                    v = False
                b[k] = v
        return b

    def merge_from_dict(self, new_dict):
        """Merge a new dict into cfg_dict.

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
    def load_from_file(filename):
        if filename is None:
            return Config()
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} not found')

        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py')
                temp_config_name = os.path.basename(temp_config_file.name)
                shutil.copyfile(filename,
                                os.path.join(temp_config_dir, temp_config_name))
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

        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)

        return Config(cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __getstate__(self):
        return self.cfg_dict

    def __setstate__(self, state):
        self.cfg_dict = state
