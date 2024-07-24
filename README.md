# Advanced Brain Tumor Classification with Explainable AI, Deep Learning, and Large Language Model Integration
*Research and implementation by Poornima Godavarthy under the guidance of Prof. Canaan*

## Description
This project aims to classify brain tumor types from the BraTs Dataset using a custom transfer learning approach based on the lightweight MobileNetV2 model. By incorporating Grad-CAM (Gradient-weighted Class Activation Mapping) for explainability and GPT-2 for generating detailed medical contexts, the project provides a comprehensive tool for brain tumor diagnosis and interpretation.

The current model demonstrates an accuracy of 86% and an F1 score of 86% on the test set, underscoring its effectiveness in accurately classifying brain tumor types. Additionally, the project enhances medical image explainability by using XAI methods to provide visual insights into the model's decision-making process, coupled with GPT-2 to produce detailed medical contexts. This integration helps healthcare professionals understand, trust, and utilize AI predictions more effectively, providing better contextual understanding for physicians.

---

<div style="text-align: center;">
    <img width="350" alt="MRI Scan" src="https://github.com/poornimagodavarthy/BrainTumor-Detection/assets/71750194/664bc1b2-021f-4826-9f54-5bb977bbe858">
    <img width="350" alt="MRI Scan" src="https://github.com/poornimagodavarthy/BrainTumor-Detection/assets/71750194/887182c5-ec65-46b6-acf1-7807d3d124a7">
</div>

## Choose How to Run
1. [On Google Colab](#running-the-notebook-on-google-colab)
2. [On Local Machine](#running-on-local-machine)
   
## Dataset
To access the dataset used in this project, click the link below:

### [**Brain Tumor Data**](https://data.mendeley.com/datasets/w4sw3s9f59/1)

## Running the Notebook on Google Colab

### Instructions:

1. **Open Google Colab:**
   - Navigate to [Colab File](https://colab.research.google.com/drive/1HYTJhRXZDrIozmVjj1ELM6LJLvCJDtJJ?usp=drive_link).

2. **Create a New Notebook:**
   - Click on File > Save a copy in Drive.

3. **Navigate to the Notebook File.**

4. **Preprocess Dataset:**
   - Download the [dataset](https://data.mendeley.com/datasets/w4sw3s9f59/1) from BraTs.
   - Preprocess by running the code blocks for testing and training.
  
5. **Getting the Model:**
   - Download and import the [model](https://drive.google.com/file/d/1-6OOWPtLyGKetNob07Fd_0zXMNjWyXEu/view?usp=sharing) to your Google Drive.
   - Mount your drive by running the code block.
     
6. **To View Grad-CAM:**
   - Load the model path and run the code block for Grad-CAM integration.

7. **Running GPT-2:**
   - Please be sure to obtain a secret access key from Hugging Face Transformers and include it in your notebook to access GPT-2.

## Running on Local Machine
Prerequisites:
- Python 3.x installed on your machine

### Instructions:

1. **Clone the Repository:**
   - Open a terminal and run the following command to clone the repository:
     
bash
     git clone https://github.com/poornimagodavarthy/BrainTumor-Detection.git


2. **Navigate to the Cloned Directory:**
   - Change to the project directory:
     
bash
     cd BrainTumor-Detection


3. **Setup Virtual Environment (Optional but Recommended):**
   - On macOS/Linux:
     
bash
     python3 -m venv myenv
     source myenv/bin/activate

   - On Windows:
     
bash
     python -m venv myenv
     .\myenv\Scripts\activate


4. **Install Dependencies:**
   - Install the required dependencies using pip:
     
bash
     pip install -r requirements.txt

   This command will install all the necessary packages listed in the requirements.txt file.
   
5. **Prepare Data (if applicable):**
   - If your project involves specific datasets or preprocessing steps (noted in preprocess.py), ensure they are set up correctly. Modify data paths or configurations as needed.

6. **Train the Model (if applicable):**
   - If training is required or you wish to customize model parameters, modify and run train.py.

7. **Evaluate the Model (if applicable):**
   - Run evaluate.py to evaluate the trained model on test data.

8. **Run Grad-CAM and GPT-2 Integration:**
   - Execute the GradCam_GPT.py script to integrate Grad-CAM visualization with GPT-2 text generation.

---

## Example Results:
<div style="text-align: center;">
    <img width="1241" alt="image" src="https://github.com/user-attachments/assets/298f284a-ca61-4e12-8849-9f7996026269">
</div>
