from .registry import Registry, get_from_name

MODEL = Registry('model')
DATASET = Registry('dataset')
PIPELINE = Registry('pipeline')
SAMPLER = Registry('sampler')


def build(cfg, registry, args=None):
    return build_from_cfg(cfg, registry, args)


def build_network(cfg):
    return build(cfg, NETWORK)


def convert_device_name(device_type, device_ids):
    """Convert device to either cpu or cuda."""
    gpu_names = ["gpu", "cuda"]
    cpu_names = ["cpu"]
    if device_type not in cpu_names + gpu_names:
        raise KeyError("the device should either "
                       "be cuda or cpu but got {}".format(device_type))
    assert type(device_ids) is list
    device_ids_new = []
    for device in device_ids:
        device_ids_new.append(int(device))

    if device_type in gpu_names:
        return "cuda", device_ids_new
    else:
        return "cpu", device_ids_new


def convert_framework_name(framework):
    """Convert framework to either tf or torch."""
    tf_names = ["tf", "tensorflow", "TF"]
    torch_names = ["torch", "pytorch", "PyTorch"]
    if framework not in tf_names + torch_names:
        raise KeyError("the framework should either "
                       "be tf or torch but got {}".format(framework))
    if framework in tf_names:
        return "tf"
    else:
        return "torch"


def get_module(module_type, module_name, framework=None, **kwargs):
    """Fetch modules (pipeline, model, or) from registry."""
    if module_type == 'pipeline':
        framework = convert_framework_name(framework)
        return get_from_name(module_name, PIPELINE, framework)

    elif module_type == "dataset":
        return get_from_name(module_name, DATASET, framework)

    elif module_type == "sampler":
        return get_from_name(module_name, SAMPLER, framework)

    elif module_type == "model":
        framework = convert_framework_name(framework)
        return get_from_name(module_name, MODEL, framework)
    else:
        raise KeyError("module type should be model, dataset, or pipeline but "
                       "got {}".format(module_type))
