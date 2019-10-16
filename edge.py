import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
DATADIR = "Products”
CATEGORIES = ["colgate"]
training data = []
IMG_SIZE = 50
def create_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category) 
class_num = CATEGORIES.index (category)
    for img in tqdm(os.listdir(path)):
      try:
img_array = cv2.imread(os.path.join(path, img), cv2. IMREAD_GRAYSCALE)
new_array = cv2.resize (img_array, (IMG_SIZE, IMG_SIZE))
traning_data. append([new_array, class_num]) 
      except Exception as  e:
          pass
create_training_data()
random.shuffle(training_data)
X=[]
y=[]
for features, label in training_data:
X.append(features)
y.append (label)
X= np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 1)

import pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_cut.close()





























from Flask import Flask, request, jsonify, render_template, send_file
import json
import base64
ihportos
from flask_cors import CORS

from predictor import predict
app = Flask (_name_, template_folder='templates')
CORS(app, resources={r" /api/*": {"origins": "*}})

@app. route('/') 
def index () :
    return render_template('index .html')

@app.route ('/assets /<path:path>')
def serve_static (path):
    return send_file ("assets /"+path)
@app.route (' /api /image_upload',methods=["POST"])
def ImageUpload():
    if "image" in (request.files):
                                  file=request.files["image"]
file.save("image.jpeg")

resp=predict()
                                  result = {
                                      'error':False,
                                      'product': resp["category"],
                                      "prediction": str(resp["prediction"])
                                  }
                                  return json.dumps(result)
if _name_ == "_main_":
app.run(host="localhost",port=5000,debug=False)






























import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keraspreprocessing.image import ImageDateGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle

NAME = "Prod-CNN"
pickle_in = open("X.pickle", "rb") 
X= pickle.load(pickle_in)

pickie_in = open("y.pickle", "rb" ) 
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()
model.add(Conv2D(256,(6,3),input_shape=X.shape[1:]))
model.add (Activation( 'relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256,(6,3)))
model.add (Activation( 'relu'))
model.add (MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 

model.add(dense(1))
model.add(Activation( 'sigmoid'))
tensorboard = Tensorboard(log_dir="logs/{}".format(NAME))

model.compile (loss='binary crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
model.fit(X, y, batch_size=24, epochs=10, validation_split=0.3)
model.save ('product-classifier.model')
























import tensorflow as tf
import cv2
from keras.models import load_model
import numpy as np

CATEGORIES =["colgate"]

def prepare(fileprt):
  IMG_SIZE = 50
img_array = cv2.imread(filepath)
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,1)

def predict():
  model = tf.keras.models.load_model("product-classifier.model")
  prediction = model.predict([prepare(r'image.jpeg')])
return{
      "prediction":rediction,
      "category":CATEGORIES[int(prediction[0][0])]
  }


