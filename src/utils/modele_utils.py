from keras import layers

def find_last_conv_layer(model):
    # On parcourt les couches à l’envers
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")