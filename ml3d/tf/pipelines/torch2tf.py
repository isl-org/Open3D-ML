import tensorflow as tf
import torch
import pudb


def torch2tf(net, path):
    wts = torch.load(path)
    wts = wts['model_state_dict']

    out_enc = []
    out_enc_non_trainable = []
    out_dec = []
    out_dec_non_trainable = []
    extra = []

    for name in wts.keys():
        full_name = name

        if 'encoder' in name:
            if 'weight' in name or 'bias' in name:
                W = wts[name].numpy()
                if 'mlp' in name:
                    W = W.transpose()
                out_enc.append((name, W))
            else:
                if 'tracked' not in name:
                    W = wts[name].numpy()
                    if 'mlp' in name:
                        W = W.transpose()
                    out_enc_non_trainable.append((name, W))

        elif 'decoder' in name:
            if 'weight' in name or 'bias' in name:
                W = wts[name].numpy()
                if 'mlp' in name:
                    W = W.transpose()
                out_dec.append((name, W))
            else:
                if 'tracked' not in name:
                    W = wts[name].numpy()
                    if 'mlp' in name:
                        W = W.transpose()
                    out_dec_non_trainable.append((name, W))
        else:
            W = wts[name].numpy()
            if 'mlp' in name:
                W = W.transpose()
            extra.append((name, W))

    out = out_enc + out_enc_non_trainable + out_dec + out_dec_non_trainable + extra

    new_wts = []
    for name, val in out:
        new_wts.append(val)

    net.set_weights(new_wts)
    print("setted")
