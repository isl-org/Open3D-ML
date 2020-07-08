from ml3d.util import Registry, build_from_cfg


NETWORK = Registry('network')
COMPOSER = Registry('composer')

def build(cfg, registry, args=None):
    return build_from_cfg(cfg, registry, args)

def build_network(cfg):
    return build(cfg, NETWORK)



