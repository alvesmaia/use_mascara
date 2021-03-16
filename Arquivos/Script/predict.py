#!/usr/bin/env python3
import io
import tensorflow as tf
from PIL import Image
import numpy as np
import argparse

def predict(image_path, model, labels):
    # load model
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # get details of model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # convert the image RGB, see input_details[0].shape
    img = Image.open(image_path).convert('RGB')

    # resize the image and convert it to a Numpy array
    img_data = np.array(img.resize(input_details[0]['shape'][1:3]))

    # run the model on the image
    interpreter.set_tensor(input_details[0]['index'], [img_data])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # get the labels
    with open(labels) as f:
        labels = f.readlines()

    return labels[np.argmax(output_data[0])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_help = True
    parser.add_argument("image", help="Path to the image to be processed")
    parser.add_argument("--model", help="Path to the model")
    parser.add_argument("--labels", help="Path to the model labels")
    args = parser.parse_args()

    assert args.image, "You must provide the image path"
    assert args.model, "You must provide the model path"
    assert args.labels, "You must provide the labels path path"
    top_prediction = predict(args.image, args.model, args.labels)
    print(top_prediction)

