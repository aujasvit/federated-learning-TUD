def ana(scale, w):
    w_attacked = copy.deepcopy(model_state_dict)
    for k in w_attacked.keys():
        noise = torch.randn(w_attacked[k].shape).to(device) * scale / 100.0 * w_attacked[k]
        w_attacked[k] = noise + w_attacked[k].float()
    return w_attacked


def sfa(scale, w):
    w_attacked = copy.deepcopy(model_state_dict)
    for k in w_attacked.keys():
        w_attacked[k] = scale * w_attacked[k].float()
    return w_attacked



