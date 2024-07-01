# Brain Tumor Classification Using XAI, Deep Learning, and LLM Integration
*Research and implementation by Poornima Godavarthy*

## Description
This project focuses on classifying brain tumor types from the BraTs Dataset using a custom transfer learning approach based on the lightweight MobileNetV2 model. By incorporating Grad-CAM (Gradient-weighted Class Activation Mapping), the project offers interpretable visualizations of the classification results. Ongoing development efforts include the integration of Large Language Models (LLMs) to enhance explainability and further improve model accuracy and the F1 score.

The current model demonstrates an accuracy of 85% and an F1 score of 85% on the test set, underscoring its effectiveness in accurately classifying brain tumor types.


<div style="text-align: center;">
    <img width="350" alt="MRI Scan" src="https://github.com/poornimagodavarthy/BrainTumor-Detection/assets/71750194/664bc1b2-021f-4826-9f54-5bb977bbe858">
    <img width="350" alt="MRI Scan" src="https://github.com/poornimagodavarthy/BrainTumor-Detection/assets/71750194/887182c5-ec65-46b6-acf1-7807d3d124a7">
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
   - Navigate to [Colab File](https://colab.research.google.com/drive/1HYTJhRXZDrIozmVjj1ELM6LJLvCJDtJJ?usp=drive_link)

2. **Create a New Notebook:**
   - Click on `File` > `Save a copy in Drive`.

3. **Navigate to the Notebook File:**
   
5. **Preprocess Dataset**
   - Download the [dataset](https://data.mendeley.com/datasets/w4sw3s9f59/1) from BraTs.
   - Preprocess by running the code blocks for testing and training.
  
6. **Getting the Model**
   - Download and import the [model](https://drive.google.com/file/d/1-6OOWPtLyGKetNob07Fd_0zXMNjWyXEu/view?usp=sharing) to your Google Drive
   - Mount your drive by running the code block
     
7. **To view Grad-Cam**
   - load the model path and run code block for grad-cam integration

## Running on Local Machine
Prerequisites:
Python 3.x installed on your machine

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

3. **Setup Virtual Environment (Optional but Recommended):**
   - On macOS/Linux:
     ```bash
     python3 -m venv myenv
     source myenv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv myenv
     .\myenv\Scripts\activate
     ```

4. **Install Dependencies:**
   - Install the required dependencies using pip:
     ```bash
     pip install -r requirements.txt
     ```
   This command will install all the necessary packages listed in the `requirements.txt` file.
   
5. **Prepare Data (if applicable):**
   - If your project involves specific datasets or preprocessing steps (noted in preprocess.py), ensure they are set up correctly. Modify data paths or configurations as needed.

6. **Train the Model (if applicable):**
   - If training is required or you wish to customize model parameters, modify and run train.py.

7. **Evaluate the Model (if applicable):**
   - Run evaluate.py to evaluate the trained model on test data.

8. **Run Grad-CAM and GPT-2 Integration:**
   - Execute the GradCam_GPT.py script to integrate Grad-CAM visualization with GPT-2 text generation.
   - Since GPT-2 integration is still under development, it is recommended to comment out that part to avoid unexpected responses.
