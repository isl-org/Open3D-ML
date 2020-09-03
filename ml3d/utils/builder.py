from .registry import Registry, get_from_name

MODEL = Registry('model')
DATASET = Registry('dataset')
PIPELINE = Registry('pipeline')



def build(cfg, registry, args=None):
    return build_from_cfg(cfg, registry, args)

def build_network(cfg):
    return build(cfg, NETWORK)

def convert_framework_name(framework):
    tf_names = ["tf", "tensorflow", "TF"]
    torch_names = ["torch", "pytorch", "PyTorch"]
    if framework not in tf_names + torch_names:
        raise KeyError(
            "the framework shoule either "
            "be tf or torch but got {}".format(framework))
    if framework in tf_names:
        return "tf"
    else:
        return "torch"


def get_module(module_type, module_name, framework=None, **kwargs):
    if module_type is 'pipeline':
        framework = convert_framework_name(framework)
        return get_from_name(module_name, PIPELINE, framework)

    elif module_type is "dataset":
        return get_from_name(module_name, DATASET, framework)

    elif module_type is "model":
        framework = convert_framework_name(framework)
        return get_from_name(module_name, MODEL, framework)
    else:
        raise KeyError(
            "module type should be model, dataset, or pipeline but "
            "got {}".format(module_type))

