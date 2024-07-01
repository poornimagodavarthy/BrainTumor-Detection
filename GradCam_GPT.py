## GRAD-CAM INTEGRATION

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Conv2D
import tensorflow as tf
import matplotlib
import transformers
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, tokenizer

#load model
model_path = '/content/drive/MyDrive/Brain Tumor data/NeuroModel.h5'
base_model = load_model(model_path)


conv_layers = [layer for layer in base_model.layers if isinstance(layer, Conv2D)]

# pick a layer to visualize
chosen_layer = conv_layers[-1].name # last conv layer


def get_img_array(img_path, size):
  img = image.load_img(img_path, target_size=size)
  array = image.img_to_array(img)
  array = np.expand_dims(array, axis=0)
  array = array / 255.0
  return array


def make_gradcam_map(img_array, layer_name, model, pred_index=None):
  grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(chosen_layer).output, model.output])
  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    if pred_index is None:
      pred_index = tf.argmax(predictions[0]) # finds class index w/ highest predicted score
    class_channel = predictions[:, pred_index]

  grads = tape.gradient(class_channel, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
  return heatmap.numpy(), np.argmax(predictions[0].numpy())

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
  img = image.load_img(img_path)
  img = image.img_to_array(img)
  heatmap = np.uint8(255 * heatmap)

  jet = matplotlib.colormaps['jet']
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  jet_heatmap = image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
  jet_heatmap = image.img_to_array(jet_heatmap)

  superimposed_img = jet_heatmap * alpha + img
  superimposed_img = image.array_to_img(superimposed_img)
  superimposed_img.save(cam_path)

  plt.imshow(superimposed_img)
  plt.axis('off')
  plt.show()


# GPT-2 Text Generation (in progress)
# comment this code to view just the grad cam integration

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tumor_types = ['glioma', 'meningioma', 'notumor', 'pituitary']


def generate_text(pred_index):
    # Get predicted tumor
    tumor_identified = tumor_types[pred_index].capitalize()
    base_prompt = {
        "Meningioma": ("A Meningioma brain tumor has been identified. \n "
                       "Meningiomas are typically benign and slow-growing tumors that originate from the meninges, \n "
                       "the protective layers surrounding the brain and spinal cord. They may cause symptoms by pressing \n "
                       "on the brain or spinal cord, leading to headaches, seizures, or neurological deficits. Treatment \n "
                       "often involves surgical removal, and the prognosis is generally favorable. \n "),
        "Glioma": ("A Glioma brain tumor has been identified. Gliomas are a type of tumor that arises from glial cells in the brain. \n"
                   "They can be benign or malignant and are categorized into different grades based on their aggressiveness. \n "
                   "Symptoms can include headaches, seizures, and neurological impairments. Treatment typically involves surgery, \n "
                   "radiation therapy, and chemotherapy."),
        "Pituitary": ("A Pituitary tumor has been identified. Pituitary tumors are abnormal growths that develop in the pituitary gland, \n "
                      "which is located at the base of the brain. These tumors can affect hormone production, leading to various endocrine disorders. \n "
                      "Symptoms may include vision problems, headaches, and hormonal imbalances. Treatment options include medication, surgery, and radiation therapy. \n"),
        "Notumor": ("No brain tumor has been identified. This could mean the observed abnormalities are not due to a tumor but might be due to other factors such as \n"
                    "inflammation, infection, or benign conditions. Further medical evaluation and imaging are recommended to determine the exact cause.\n")
    }
    input_text = base_prompt.get(tumor_identified, "No information available for this type of tumor.")

    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the text with attention mask
    generated_text = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=90,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return output_text


#display grad cam
img_path = 'path/to/image'
img_array = get_img_array(img_path, (512, 512))

heatmap, pred_index = make_gradcam_map(img_array, chosen_layer, base_model)

print("Tumor Type:", tumor_types[pred_index].capitalize())
display_gradcam(img_path, heatmap)


# Generate text based on the predicted tumor type (in progress)
output_text = generate_text(pred_index)
print(output_text)