"""Evaluation script for both CapsNet and CNN models.

Example use:

    $ python evaluation/test.py --dataset=fer --network-type=capsnet_single_digit_capsule --weights=./weights-01.h5

"""
import argparse

import numpy as np
from tensorflow import keras

from CapsNet_vs_CNN.dataset import fer2013_3

CAPSNET_ROUTINGS = 3        # TODO: Allow to override it!
ROTATIONS = [-180, -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15,
             0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]


def evaluate(dataset, network_type, weights):
    if dataset == 'fer':
        from CapsNet_vs_CNN.dataset import fer2013_3
        dataset_module = fer2013_3
    else:
        print('ERROR! Unsupported dataset!')
        exit(1)

    if 'capsnet' in network_type:
        if dataset == 'fer' and network_type == 'capsnet_single_digit_capsule':
            from CapsNet_vs_CNN.capsnet.singledigitcapsule.fer_capsulenet_3 import CapsNet
            model, _, _ = CapsNet(input_shape=fer2013_3.IMAGE_SHAPE, n_class=fer2013_3.CLASSES, routings=CAPSNET_ROUTINGS)
        elif dataset == 'fer' and network_type == 'capsnet_two_digits_capsules':
            from CapsNet_vs_CNN.capsnet.twodigitcapsules.fer_capsulenet_3 import CapsNet
            model, _, _ = CapsNet(input_shape=fer2013_3.IMAGE_SHAPE, n_class=fer2013_3.CLASSES, routings=CAPSNET_ROUTINGS)
        else:
            print('ERROR! Unsupported CapsNet type!')
            exit(1)

    elif network_type == 'cnn':
        if dataset == 'fer':
            from CapsNet_vs_CNN.cnn.fer_resnet_3 import resnet_v2
            model = resnet_v2(input_shape=fer2013_3.IMAGE_SHAPE, depth=110)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.1), metrics=['accuracy'])
    else:
        print('ERROR! Unsupported Network Type!')
        exit(1)

    model.summary()
    model.load_weights(weights)

    for rotation in ROTATIONS:
        if 'capsnet' in network_type:
            x_test, y_test = dataset_module.get_test_data_for_capsnet(rotation)
            y_pred, x_reconstructed = model.predict(x_test, batch_size=100)
            accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test[0], 1))/y_test[0].shape[0]
        elif network_type == 'cnn':
            x_test, y_test = dataset_module.get_test_data_for_cnn(rotation)
            scores = model.evaluate(x_test, y_test, verbose=1)
            accuracy = scores[1]

        print('Rotation:', rotation,
              'Accuracy:', accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of CNN and CapsNet networks against rotated images.')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--network-type', required=True, type=str)
    parser.add_argument('--weights', required=True, type=str)
    args = parser.parse_args()
    print('args:', args)

    evaluate(args.dataset, args.network_type, args.weights)

