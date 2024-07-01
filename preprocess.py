import os
import cv2
from PIL import Image

# preprocess training set and save 
def preprocess_training_images(training_set, target_set, target_size=(512, 512)):
  if not os.path.exists(target_set):
        os.makedirs(target_set)

  preprocessed_data = []
  tumor_types = ["glioma", "meningioma", "notumor", "pituitary"]
  for tumor_type in tumor_types:
    type_path = os.path.join(training_set, tumor_type)
    target_type_path = os.path.join(target_set, tumor_type)

    if not os.path.exists(target_type_path):
            os.makedirs(target_type_path)

    if os.path.isdir(type_path):
      print(f"Processing directory: {type_path}")

      for image_name in os.listdir(type_path):
        image_path = os.path.join(type_path, image_name)
        target_image_path = os.path.join(target_type_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          pil_image = Image.fromarray(image_rgb)
          resized_image = pil_image.resize((512, 512))
          resized_image.save(target_image_path)
          print(f"Saved image: {target_image_path}")
        else:
          print(f"Failed to read image: {image_path}")

# preprocess test set and save to drive
def preprocess_test_images(test_set, target_set, target_size=(512, 512)):
  if not os.path.exists(target_set):
        os.makedirs(target_set)

  preprocessed_data = []
  tumor_types = ["glioma", "meningioma", "notumor", "pituitary"]
  for tumor_type in tumor_types:
    type_path = os.path.join(test_set, tumor_type)
    target_type_path = os.path.join(target_set, tumor_type)

    if not os.path.exists(target_type_path):
            os.makedirs(target_type_path)

    if os.path.isdir(type_path):
      print(f"Processing directory: {type_path}")

      for image_name in os.listdir(type_path):
        image_path = os.path.join(type_path, image_name)
        target_image_path = os.path.join(target_type_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          pil_image = Image.fromarray(image_rgb)
          resized_image = pil_image.resize((512, 512))
          resized_image.save(target_image_path)
          print(f"Saved image: {target_image_path}")
        else:
          print(f"Failed to read image: {image_path}")


training_set = '/path/to/your/training/directory'
resized_training_set = '/path/to/resized/training/images'
preprocess_training_images(training_set, resized_training_set, (512, 512))

test_set = '/path/to/your/testing/directory'
resized_test_set = '/path/to/resized/testing/images'
preprocess_test_images(test_set, resized_test_set, (512, 512))