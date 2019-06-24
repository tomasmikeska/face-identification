'''
Convert trained model to ready-to-use model

Model used for training contained 2 inputs and output in order to use it
with Center loss layers. This module converts it into model with single
input and single normalized embedding output.
'''
import argparse
from keras.models import Model
from model import load_model
from constants import INPUT_SHAPE, EMBEDDING_SIZE


EMB_LAYER_INDEX = -4


def convert_to_prod_model(model):
    prod_model = Model(inputs=model.inputs[0], outputs=model.layers[EMB_LAYER_INDEX].output)
    prod_model.compile(optimizer='nadam', loss='categorical_crossentropy')
    return prod_model


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Convert centerloss model to production structure model')
    parser.add_argument('--weights', type=str, help='Trained model weights path')
    parser.add_argument('--nclasses', type=int, help='Number of output classes during training (printed on start)')
    parser.add_argument('-o', '--output', type=str, help='Output model save path - structure + weights')
    args = parser.parse_args()
    # Convert
    model = load_model(INPUT_SHAPE, args.nclasses, EMBEDDING_SIZE)
    model.load_weights(args.weights)
    prod_model = convert_to_prod_model(model)
    prod_model.save(args.output)
    print('Success!')
