# Custom Drawing Classifier
This is a custom drawing classifier application that allows users to draw simple shapes or numbers and classify them 
into different classes using various machine learning models. The application provides a graphical user interface (GUI) 
built with tkinter for easy interaction.

## Features
**Drawing Canvas:** Users can draw shapes using the left mouse button on the canvas. The drawn image is displayed in real-time.

**Brush Size:** Users can adjust the brush size by clicking the "Brush-" or "Brush+" buttons.

**Class Selection:** Users can assign a class to the drawn shape by clicking the corresponding class button.

**Model Training:** Users can train the classification model by using the "Train Model" button. The application extracts the 
training data from the saved images in the respective class directories and trains the model using different machine learning 
algorithms.

**Model Prediction:** Users can predict the class of a drawn shape by clicking the "Predict" button. The application saves the 
drawn image, preprocesses it, and feeds it to the trained model for prediction.

**Model Rotation:** Users can switch between different machine learning models by clicking the "Change Model" button. 
The available models include:
Support Vector Machine (SVM)
Gaussian Naive Bayes
Decision Tree
K Neighbors Classifier
Random Forest
Logistic Regression

**Model Saving and Loading:** Users can save the trained model to a file using the "Save Model" button and load a previously 
saved model using the "Load Model" button.

**Project Saving:** Users can save the entire project, including class names, class directories, model, and project metadata, 
using the "Save Everything" button.

**Project Initialization:** If the project directory already exists, the application loads the previous project data, including 
class names, counters, and the trained model. Otherwise, the application prompts the user to enter class names and initializes 
the project.

## Requirements
The following libraries are required to run the application:

* tkinter
* PIL (Python Imaging Library)
* OpenCV (cv2)
* numpy
* scikit-learn (sklearn

## Usage
* Clone or download the repository to your local machine.
* Install the required libraries mentioned in the "Requirements" section.
* Run the *drawing_classifier.py* script to start the application.
* Follow the prompts to enter the project name and class names if it's a new project.
* Use the GUI to draw shapes, assign classes, train the model, predict shapes, switch models, save/load models, and save the 
entire project.

## User Interface Preview
* Defining the Project name and respective Classes

![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/0e054850-93c6-419d-a905-f2539ef0292d)

![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/964172a7-05e5-4847-87be-d2e7ba6c82c4)
![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/ec28f985-8136-473d-a3c7-289a02916085)
![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/57ada327-625b-4535-8725-7ecdacf415f6)

* User Interface for training the data

![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/22ab2445-3bd7-4d4e-9c32-dacea419aeb1)

* Message printed when the training is complete

![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/5a8fc32a-979b-428b-a761-df22c45df561)

* Message printing the probable answer is received when predicted

![image](https://github.com/ShakalyaGarg/CanvasAnalyzer/assets/129611852/2bd374b0-d9b0-4b9b-b113-2ccba1836cbf)

## Machine Learning Models
The Custom Drawing Classifier application uses the following machine learning models for shape classification:

**Support Vector Machine (SVM):** A supervised learning model that analyzes data and recognizes patterns used for classification
and regression analysis.

**Gaussian Naive Bayes:** A classification algorithm based on Bayes' theorem with the assumption of independence between 
features.

**Decision Tree:** A flowchart-like structure in which each internal node represents a feature, each branch represents a 
decision rule, and each leaf node represents the outcome.

**K Neighbors Classifier:** A non-parametric classification algorithm that classifies objects based on the majority vote of 
their neighbors.

**Random Forest:** An ensemble learning method that constructs multiple decision trees and outputs the class that is the mode 
of the classes predicted by individual trees.

**Logistic Regression:** A statistical model that uses a logistic function to model a binary dependent variable.

## Repository Structure
The repository is structured as follows:

* ***drawing_classifier.py:*** The main Python script that contains the Custom Drawing Classifier application code.
* **README.md:** The README file with project documentation.
* **project_name/:** The main project directory (replace project_name with your chosen project name).
    * class1/: Directory for images of class 1 shapes.
    * class2/: Directory for images of class 2 shapes.
    * class3/: Directory for images of class 3 shapes.
    * project_name_data.pickle: Pickle file that stores project metadata, including class names, counters, and the trained model.
