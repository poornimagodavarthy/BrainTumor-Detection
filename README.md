# Brain Tumor Detection with XAI and Deep Learning
*Research by Poornima Godavarthy*

## Description
This project aims to classify tumor types from the BraTs Dataset utilizing a custom transfer learning approach based on the lightweight MobileNetV2 model. By applying Grad-CAM (Gradient-weighted Class Activation Mapping), the project provides interpretable visualizations of the classification results. Ongoing work includes integrating Large Language Models (LLMs) to enhance explainability and improve model accuracy.

<div style="text-align: center;">
    <img width="293" alt="MRI Scan" src="https://github.com/poornimagodavarthy/BrainTumor-Detection/assets/71750194/664bc1b2-021f-4826-9f54-5bb977bbe858">
    <img width="273" alt="MRI Scan" src="https://github.com/poornimagodavarthy/BrainTumor-Detection/assets/71750194/b4649318-06e3-475c-9a1e-29561ea3566f">
</div>




## Choose how to run
1. [On Google Colab](#running-the-notebook-on-google-colab)
2. [On Local Machine](#running-on-local-machine)
   
## Dataset
To access the dataset used in this project, click the link below:

### [**Brain Tumor Data**](https://data.mendeley.com/datasets/w4sw3s9f59/1)

## Running the Notebook on Google Colab

### Instructions:

1. **Open Google Colab:**
   - Navigate to [Colab File](https://colab.research.google.com/drive/1bM4zTYlEN_mPA8C5Z6a2jfTv1dNsMduy#scrollTo=CUKvWrHOJAng)

2. **Create a New Notebook:**
   - Click on `File` > `Save a copy in Drive`.

3. **Navigate to the Notebook File:**
   
5. **Preprocess Dataset**
   - Download the [dataset](https://data.mendeley.com/datasets/w4sw3s9f59/1) from BraTs.
   - Preprocess by running the code blocks for testing and training.
  
6. **Getting the Model**
   - Download and import the [model](https://drive.google.com/file/d/1Gn3KVwo0jEgyY6F3_-I3UfeCo-2gyJzZ/view?usp=drive_link) to your Google Drive
   - Mount your drive by running the code block
     
7. **To view Grad-Cam**
   - load the model path and run code block for grad-cam integration

## Running on Local Machine


### Instructions:

1. **Clone the Repository:**
   - Open a terminal and run the following command to clone the repository:
     ```bash
     git clone https://github.com/poornimagodavarthy/BrainTumor-Detection.git
     ```

2. **Navigate to the Cloned Directory:**
   - Change to the project directory:
     ```bash
     cd BrainTumor-Detection
     ```

3. **Install Dependencies:**
   - Install the required dependencies using pip:
     ```bash
     pip install -r requirements.txt
     ```

   This command will install all the necessary packages listed in the `requirements.txt` file.


