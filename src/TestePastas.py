import os
import numpy as np
import keras
from tensorflow.keras.utils import load_img, img_to_array

root_folder_path = "Test"
class_names = ["Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___healthy"]
img_width, img_height = 256, 256

model = keras.models.load_model('Modelos\Dados12023-04-14 05_53_03.495761.keras')

correct_predictions = 0
error_predictions = 0
total_predictions = 0

def correct(prediction_class_index, real_class_index):
    if prediction_class_index == real_class_index:
      correct_predictions += 1
    else:
      error_predictions += 1
    total_predictions += 1


for folder_name in os.listdir(root_folder_path):
    folder_path = os.path.join(root_folder_path, folder_name)
    class_index = class_names.index(folder_name)
    
    for filename in os.listdir(folder_path):
      img_path = os.path.join(folder_path, filename)
      img = load_img(img_path, target_size=(img_width, img_height))
      img = img.resize((256,256))
      test_image = img_to_array(img)
      test_image = test_image / 255.0
      img_array = np.array(test_image)
      test_image = np.expand_dims(test_image, axis=0)
      prediction = model.predict(test_image)
      predicted_class_index = np.argmax(prediction)
      if predicted_class_index == class_index:
        correct_predictions += 1
      else:
        error_predictions += 1
      total_predictions += 1
      print("pasta: {} arquivo: {}".format(folder_name,filename))
      print("previsão: {}".format(class_names[predicted_class_index]))
      print("total de previsões = {} - previsões corretas = {} - previsões erradas = {}"
      .format(total_predictions,
      correct_predictions,
      error_predictions))

accuracy = correct_predictions / total_predictions
print("Precisão do modelo: {:.2f}%".format(accuracy * 100))