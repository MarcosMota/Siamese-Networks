from typing import Tuple
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def initialize_base_network(shape: Tuple[int,int,int]) -> Model:
    input = Input(shape=shape, name="base_input")
    x = Flatten(name="flatten_input")(input)
    
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    
    x = Dense(128, activation='relu', name="third_base_dense")(x)
    return Model(inputs=input, outputs=x)

def build_model(shape: Tuple[int,int,int]) -> Model:

    base_network = initialize_base_network()
    
    input_a = Input(shape=shape, name="left_input")
    vect_output_a = base_network(input_a)

    input_b = Input(shape=shape, name="right_input")
    vect_output_b = base_network(input_b)

    output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

    return Model([input_a, input_b], output)

