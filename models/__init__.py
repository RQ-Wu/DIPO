models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(name, config):
    if name == 'model_B9':
        name = 'denoiser_singapo'
    model = models[name](config)
    return model


from . import denoiser