from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def linear_deep_grappa_model(ncoils=15):
    model = Sequential([Dense(ncoils, use_bias=False, activation='linear')])
    return model
