import cv2
from PIL import Image
from tensorflow import keras
import numpy as np
from flask import Flask, request, jsonify
import werkzeug

app = Flask(__name__)


def predicter(path, cnn_model):
    animals_list = [
        'Close_Eyes',
         'Open_Eyes'
    ]
    Image_Shape = (224, 224)
    path = path
    animal = Image.open(path).resize(Image_Shape)
    # animal = np.array(animal) / 255.0
    # animal = animal[np.newaxis, ...]
    # result = cnn_model.predict(animal)
    image = cv2.imread(str(path))
    if image is None:
        print("Wrong image")
        return "Golden_Jackal"
    else:
        image_resized = cv2.resize(image, (224, 224))
        image = np.expand_dims(image_resized, axis=0)
        pred = cnn_model.predict(image)
        output_class = animals_list[np.argmax(pred)]
        return output_class



@app.route('/upload', methods=["POST"])
def upload():  # put application's code here
    cnn_model = keras.models.load_model(
        'F:/4th Smester/work/drowziness model/detection.h5')

    if (request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploadedimages/" + filename)
        path_of_image = "./uploadedimages/" + filename
        animal_name = predicter(path_of_image, cnn_model)
        return jsonify({
            "message": animal_name
        })



if __name__ == '__main__':
    app.run(debug=True, port=5000)