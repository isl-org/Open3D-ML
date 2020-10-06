import tensorflow as tf
import torch
import pudb
import numpy as np
import os


def tf2tf(net, path):
    path = os.path.abspath(path)
    init_vars = tf.train.list_variables(path)
    tf_vars = []
    for name, shape in init_vars:
        arr = tf.train.load_variable(path, name)
        tf_vars.append((name, arr))

    # for name, arr in tf_vars:
    #     print(name, arr.shape)
    # # print(net.get_weights())

    # for i, arr in enumerate(net.weights):
    #     print(i, arr.name, arr.shape)
    # print(len(net.weights))

    # exit(0)

    head = []
    head_non_trainable = []
    out_enc = []
    out_enc_non_trainable = []
    out_dec = []
    out_dec_non_trainable = []
    extra = []

    for name, arr in tf_vars:
        full_name = name

        if '/layer' in name:
            if 'beta' in name or 'gamma' in name or 'weights' in name:
                if 'simple_0' in name:
                    if 'weights' not in name:
                        head.append((name, arr))
                    else:
                        prev = head[-1]
                        prev_prev = head[-2]
                        if 'beta' in prev_prev[0] and 'gamma' in prev[0]:
                            head = head[:-2]
                            head.append((name, arr))
                            head.append(prev_prev)
                            head.append(prev)
                        else:
                            head.append((name, arr))
                else:
                    if 'weights' not in name:
                        out_enc.append((name, arr))
                    else:
                        prev = out_enc[-1]
                        prev_prev = out_enc[-2]
                        if 'beta' in prev_prev[0] and 'gamma' in prev[0]:
                            out_enc = out_enc[:-2]
                            out_enc.append((name, arr))
                            out_enc.append(prev_prev)
                            out_enc.append(prev)
                        else:
                            out_enc.append((name, arr))
            else:
                if 'simple_0' in name:
                    if 'kernel_points' not in name:
                        head_non_trainable.append((name, arr))
                    else:
                        prev = head_non_trainable[-1]
                        prev_prev = head_non_trainable[-2]
                        if 'moving_mean' in prev_prev[
                                0] and 'moving_variance' in prev[0]:
                            head_non_trainable = head_non_trainable[:-2]
                            head_non_trainable.append((name, arr))
                            head_non_trainable.append(prev_prev)
                            head_non_trainable.append(prev)
                        else:
                            head_non_trainable.append((name, arr))
                else:
                    if 'kernel_points' not in name:
                        out_enc_non_trainable.append((name, arr))
                    else:
                        prev = out_enc_non_trainable[-1]
                        prev_prev = out_enc_non_trainable[-2]
                        if 'moving_mean' in prev_prev[
                                0] and 'moving_variance' in prev[0]:
                            out_enc_non_trainable = out_enc_non_trainable[:-2]
                            out_enc_non_trainable.append((name, arr))
                            out_enc_non_trainable.append(prev_prev)
                            out_enc_non_trainable.append(prev)
                        else:
                            out_enc_non_trainable.append((name, arr))
        elif 'uplayer' in name:
            if 'beta' in name or 'gamma' in name or 'weights' in name:
                out_dec.append((name, arr))
            else:
                out_dec_non_trainable.append((name, arr))

        else:
            extra.append((name, arr))

    out_dec = list(out_dec[i] for i in [11, 9, 10, 8, 6, 7, 5, 3, 4, 2, 0, 1])
    out_dec_non_trainable = list(
        out_dec_non_trainable[i] for i in [6, 7, 4, 5, 2, 3, 0, 1])
    extra = list(extra[i] for i in [4, 0, 1, 2, 3, 6, 5])

    out = head + out_enc + head_non_trainable + out_enc_non_trainable + out_dec + out_dec_non_trainable + extra

    for i, (name, arr) in enumerate(out):
        if 'gamma' in name:
            gamma = out[i]
            beta = out[i - 1]
            out[i] = beta
            out[i - 1] = gamma

    # for i, (name, arr) in enumerate(out):
    #     print(i, name, arr.shape)

    # print(len(out))
    # exit(0)

    new_wts = []
    for name, val in out:
        new_wts.append(val)

    net.set_weights(new_wts)
    print("setted")
