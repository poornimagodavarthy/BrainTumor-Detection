# Brain Tumor Detection with XAI and Deep Learning

## Description
This project aims to classify tumor types from the BraTs Dataset utilizing a custom transfer learning approach based on the lightweight MobileNetV2 model. 
By applying Grad-CAM (Gradient-weighted Class Activation Mapping), the project provides interpretable visualizations of the classification results. 
Ongoing work includes integrating Large Language Models (LLMs) to enhance explainability and improve model accuracy.

## Running the Notebook on Google Colab

### Instructions:

1. **Clone the Repository:**
   - Open the repository on GitHub.
   - Click the green `Code` button and copy the URL provided (e.g., `https://github.com/yourusername/brain-tumor-detection.git`).

2. **Open Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/).

3. **Create a New Notebook:**
   - Click on `File` > `New notebook`.

4. **Clone the GitHub Repository in Colab:**
   - In the first code cell of your new notebook, enter the following command to clone your repository:
     ```python
     !git clone https://github.com/yourusername/brain-tumor-detection.git
     ```
   - Run the cell by pressing `Shift + Enter`.

5. **Navigate to the Notebook File:**
   - After cloning the repository, navigate to your notebook file. For example:
     ```python
     %cd brain-tumor-detection
     ```

6. **Open the Notebook:**
   - Open your notebook file within Colab by clicking `File` > `Open notebook` > `GitHub` and then search for your repository and select your notebook file.
