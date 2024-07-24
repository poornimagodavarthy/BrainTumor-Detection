## GRAD-CAM INTEGRATION

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Conv2D
import tensorflow as tf
import matplotlib
import warnings
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


# GPT-2 Text Generation 
  
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
    "Meningioma": (
        "A Meningioma brain tumor has been identified in the patient's brain scan. Meningiomas are typically benign and slow-growing tumors that originate from the meninges, \n"
        "the protective layers surrounding the brain and spinal cord. "
        "Although generally benign, they can cause symptoms by pressing on the brain or spinal cord. \n"
        "Symptoms:\nMeningiomas may lead to headaches, seizures, vision problems, or other neurological issues due to their pressure on surrounding structures.\n\n"
        "Treatment Options:\nSurgery: Often the primary treatment, aiming to remove the tumor while preserving neurological function.\n"
        "Radiation Therapy: Used when surgery is not feasible or to target residual tumor cells.\n\n"
        "Prognosis:\nThe prognosis for meningioma patients is generally favorable, especially if the tumor is completely removed. Regular follow-up is essential to monitor for recurrence.\n\n"
        "Action Steps:\nFurther Evaluation: Additional imaging or biopsy may be required to assess the tumor's characteristics and plan treatment.\n"
        "Treatment Planning: Work with a multidisciplinary team to develop an individualized treatment strategy based on the tumor's size, location, and the patient's overall health.\n"
        "Patient Support: Provide resources and support for the patient, including information on treatment options and potential side effects."
    ),

    "Glioma": (
        "A Glioma brain tumor has been identified in the patient's brain scan. Gliomas are tumors that originate from the glial cells, which support and protect neurons in the brain. \n"
        "They can be benign or malignant and are categorized into different grades based on their aggressiveness:\n\n"
        "Grade I: Typically slow-growing and less aggressive; treatment often involves surgical removal.\n"
        "Grade II: More likely to spread; may require additional treatments such as radiation therapy.\n"
        "Grade III: Malignant tumors that grow rapidly and may infiltrate surrounding brain tissue; treatment usually involves surgery, radiation, and chemotherapy.\n"
        "Grade IV: Highly aggressive with a poor prognosis; requires aggressive treatment strategies, including surgery, radiation, chemotherapy, and possibly clinical trials for experimental therapies.\n\n"
        "Symptoms:\nPatients with gliomas may experience persistent headaches, seizures, or neurological deficits, including changes in vision, speech, or motor functions. \n"
        "The severity and type of symptoms often correlate with the tumor's location and grade.\n\n"
        "Treatment Options:\nSurgery: Aims to remove as much of the tumor as possible while preserving brain function.\n"
        "Radiation Therapy: Used to target and destroy remaining tumor cells after surgery or when surgery is not feasible.\n"
        "Chemotherapy: Utilizes drugs to kill or inhibit tumor cells, often used in conjunction with surgery and radiation.\n\n"
        "Prognosis:\nThe prognosis for glioma patients depends on the tumor's grade, location, and the patient’s overall health. Early detection and a comprehensive treatment plan can improve outcomes.\n\n"
        "Action Steps:\nFurther Evaluation: Consider additional imaging or biopsy if not already performed to determine the exact grade and extent of the tumor.\n"
        "Treatment Planning: Collaborate with a multidisciplinary team including neurosurgeons, oncologists, and radiologists to create a personalized treatment plan.\n"
        "Patient Support: Provide support resources for the patient, including information on treatment options and potential side effects."
    ),

    "Pituitary": (
        "A Pituitary tumor has been identified in the patient's brain scan. Pituitary tumors are abnormal growths that develop in the pituitary gland, which is located at the base of the brain. "
        "These tumors can disrupt hormone production and lead to a range of symptoms, including vision problems, headaches, and hormonal imbalances. Pituitary tumors can be classified based on their hormonal activity and location within the gland:\n\n"
        "Functional Pituitary Tumors: Secrete excess hormones, causing conditions such as Cushing's disease, acromegaly, or prolactinoma.\n"
        "Non-Functional Pituitary Tumors: Do not secrete hormones but can cause symptoms by pressing on nearby structures, including the optic nerves.\n\n"
        "Symptoms:\nCommon symptoms include visual disturbances, headaches, and hormonal imbalances such as abnormal growth, metabolism, or reproductive functions.\n\n"
        "Treatment Options:\nSurgery: Aims to remove the tumor, often performed through a transsphenoidal approach.\n"
        "Medication: Used to control hormone levels and manage symptoms, particularly for functional tumors.\n"
        "Radiation Therapy: May be considered if surgery is not fully effective or if the tumor is not operable.\n\n"
        "Prognosis:\nPrognosis varies depending on the tumor type, size, and treatment response. Many patients have favorable outcomes with appropriate management, but ongoing follow-up is important.\n\n"
        "Action Steps:\nFurther Evaluation: Additional imaging or testing may be needed to assess the tumor's characteristics and impact on hormone levels.\n"
        "Treatment Planning: Collaborate with endocrinologists and neurosurgeons to develop a personalized treatment plan.\n"
        "Patient Support: Offer resources and support related to managing symptoms, understanding treatment options, and addressing any side effects."
    ),

    "No Tumor": (
        "No brain tumor has been identified in the patient's scan. This result suggests that the observed abnormalities are not due to a tumor but may be related to other factors such as inflammation, infection, or benign conditions. "
        "Further medical evaluation is recommended to determine the exact cause of the abnormalities. This may include additional imaging, tests, or consultations with specialists to provide a comprehensive diagnosis and appropriate care plan.\n\n"
        "Action Steps:\nFurther Evaluation: Consider additional diagnostic tests or imaging studies to clarify the nature of the abnormalities observed.\n"
        "Consultation: Follow up with healthcare providers to explore any symptoms or concerns and determine the next steps for diagnosis or treatment.\n"
        "Patient Support: Provide information and support for the patient regarding potential causes of their symptoms and options for further evaluation."
    )
}
    input_text = base_prompt.get(tumor_identified, "No information available for this type of tumor.")
    input_ids = tokenizer(input_text, return_tensors="pt")['input_ids']
    inputs = tokenizer(input_text, return_tensors="pt")


    # Generate the text with attention mask
    generated_text = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length = len(input_ids[0])+1,
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
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

# Generate context summary based on the predicted tumor type
output_text = generate_text(pred_index)
print(output_text)