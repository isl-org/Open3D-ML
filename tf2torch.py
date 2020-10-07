import tensorflow as tf
import torch
import pudb
import numpy as np
import os


tf_path = './converted_wts/KPFCNN_Toronto3D_tf/checkpoint/ckpt-1'
init_vars = tf.train.list_variables(tf_path)
tf_vars = []
for name, shape in init_vars:
    arr = tf.train.load_variable(tf_path, name)
    name = name.replace('net/', '')
    name = name.replace('/', '.')
    name = name.replace('..ATTRIBUTES.VARIABLE_VALUE', '')
    name = name.replace('kp_conv_W', 'weights')
    name = name.replace('kp_conv_1_W', 'weights')
    name = name.replace('kp_conv_2_W', 'weights')
    name = name.replace('kp_conv_3_W', 'weights')
    name = name.replace('kp_conv_4_W', 'weights')
    name = name.replace('kp_conv_5_W', 'weights')
    name = name.replace('kp_conv_6_W', 'weights')
    name = name.replace('kp_conv_7_W', 'weights')
    name = name.replace('kp_conv_8_W', 'weights')
    name = name.replace('kp_conv_9_W', 'weights')
    name = name.replace('gamma', 'weight')
    name = name.replace('beta', 'bias')
    name = name.replace('moving_mean', 'running_mean')
    name = name.replace('moving_variance', 'running_var')
    if 'kernel_points' not in name:
        name = name.replace('kernel', 'weight')
    name = name.replace('batch_norm_block_38_b', 'bias')

    tf_vars.append((name, arr))

torch_v = torch.load('./logs/KPFCNN_Toronto3D_torch/checkpoint/temp.pth')

# for name, arr in tf_vars:
#     try:
#         print(name, arr.shape)
#     except:
#         pass
#         # print(name, arr)

torch_vars = torch_v['model_state_dict']
# print(len(torch_vars))

# for key in torch_vars.keys():
#     print(key, torch_vars[key].shape)

dict_tf = {}
for name, arr in tf_vars:
        dict_tf[name] = arr

dict_wts = {}
for key in torch_vars.keys():
    if 'tracked' in key:
        continue
    if 'mlp.weight' in key:
        dict_wts[key] = torch.Tensor(dict_tf[key].transpose())
    else:
        dict_wts[key] = torch.Tensor(dict_tf[key])


# for key in torch_vars.keys():
#     if 'tracked' in key:
#         continue
#     print(key, torch_vars[key].shape, dict_wts[key].shape)

# print(torch_v['model_state_dict'])
# print(dict_wts)
# exit(0)

torch_wts = {
    'epoch': 1,
    'model_state_dict': dict_wts,
    'optimizer_state_dict': torch_v['optimizer_state_dict'],
    'scheduler_state_dict': torch_v['scheduler_state_dict']
}
torch.save(torch_wts, './converted_wts/KPFCNN_Toronto3D_torch/wts.pth')