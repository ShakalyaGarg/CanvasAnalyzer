import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np

#MODELS USED
from sklearn.svm import LinearSVC                       #Support Vector Machine
from sklearn.naive_bayes import GaussianNB              #Gausian Naive Bayes
from sklearn.tree import DecisionTreeClassifier          #Decision Tree
from sklearn.neighbors import KNeighborsClassifier      #K Neighbors Classifier
from sklearn.ensemble import RandomForestClassifier     #Random Forest
from sklearn.linear_model import LogisticRegression     #Logistic Regression


class DrawingClassifier:

    ### MAIN CONSTRUCTOR ###
    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None     #Different classes that we will be defining

        # Counter for the drawings in the respective classes used to train the models
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None

        self.clf = None     #Model used for training and prediction
        self.proj_name = None     #Name of the directory
        self.root = None    #Used for tkinter window
        self.image1 = None    #Take the canvas and convert it into an image that can be feeded to the model

        self.status_label = None     #Used by tkinter to keep track of current model being used
        self.canvas = None
        self.draw = None      #object from PIL so that we can actually draw on the canvas

        self.brush_width = 15     #default brush size

        self.classes_prompt()
        self.init_gui()

    ### Defining/Initializing the Classes ###
    def classes_prompt(self):
        msg = Tk()     #tkinter object
        msg.withdraw     #we need a window that we can work with but not necessarily display it

        #Asking the user for the project name
        self.proj_name = simpledialog.askstring("Project Name","Please enter your project name down below",parent=msg)

        #we check if the project been created or not and if it is we will create a pickle file to store all the info/metadata regarding the project
        #if it hasnt we will create new classes instead
        if os.path.exists((self.proj_name)):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle","rb") as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']
            self.clf = data['clf']
            self.proj_name = data['pname']

        else:
            self.class1 = simpledialog.askstring("Class 1","What is the first class name",parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class name", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third class name", parent=msg)

            #initializing the counters to 1
            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1

            self.clf = LinearSVC()      #default classifier

            os.mkdir(self.proj_name)    #create the project directory
            os.chdir(self.proj_name)    #go in the class directory
            os.mkdir(self.class1)       #define the directory for classes
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")              #go back to the main directory

    ### Initialize Graphical User Interface ###
    def init_gui(self):
        WIDTH = 500         #basic dimensions
        HEIGHT = 500
        WHITE = (255,255,255)     #rgb color scheme

        self.root = Tk()
        self.root.title(f"Custom Drawing Classifier - {self.proj_name}")

        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")     #creating the custom canvas
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>",self.paint)      #binding left mouse button to paint function
        #whenever the left mouse button is clicked we will access the paint function and will be able to write in the image section of the base canvas


        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)     #image section of the canvas
        self.draw = PIL.ImageDraw.Draw(self.image1)         #To draw the image

        btn_frame = tkinter.Frame(self.root)     #defining the button frame
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)      #initializing column canvas with equal weight that can be used for buttons
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        # metadata about the Class buttons, it uses the save function to save the class for the drawn image
        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        # metadata about the Brush buttons, it uses the functions to increase, decrease the brush size and clear the image section
        bm_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        bp_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W + E)

        # metadata about the Current Model specfications, train, load or save
        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)

        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        #metadata to Change btw diff Models, Predict the output and saving the whole project
        change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)

        save_everything_btn = Button(btn_frame, text="Save Everything ", command=self.save_everything)
        save_everything_btn.grid(row=3, column=2, sticky=W + E)

        #metadata about the Status Label
        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial",10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        #when window is closed, we assign another function to close the gui
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost",True)
        self.root.mainloop()

    ### Function to Initialize the Mouse Movement when Left Mouse Button is clicked ###
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)      #initializing the canvas for drawing
        #DRawing on the canvas
        self.draw.rectangle([x1, y1, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    ### Function to Save the Image in the Canvas in a file and that file should be put into the Directory of the Respective Class with the Respective Counter Number ###
    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50),PIL.Image.ANTIALIAS)

        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png","PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png","PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.proj_name}/{self.class3}/{self.class3_counter}.png","PNG")
            self.class3_counter += 1

        #whenever we save something we are clearing the canvas there itself
        self.clear()

    ### Function to Decrease the Size of the Brush ###
    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    ### Function to Clear the Canvas ###
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white") #done by replacing the whole thing with new white

    ### Function to Increase the Size of the Brush ###
    def brushplus(self):
        self.brush_width += 1

    ### Function used to Extract the Training Data and Train the Model ###
    def train_model(self):
        img_list = np.array([])    #xtrain
        class_list = np.array([])     #ytrain

        #Getting the training data from the images saved in different directories of different classes
        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:,:,0]
            img = img.reshape(2500)    #size of the image will be 2500 as the pixels taken are 50x50
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.proj_name}/{self.class3}/{x}.png")[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(self.class1_counter - 1 + self.class2_counter - 1 + self.class3_counter - 1, 2500)

        #Training the model
        self.clf.fit(img_list, class_list)
        tkinter.messagebox.showinfo("Custom Drawings Classifier", "Model Successfully Trained!", parent=self.root)

    ### Function to Predict the Image Drawn ###
    def predict(self):
        #Extracting the image
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50), PIL.Image.ANTIALIAS)
        img.save("predictshape.png","PNG")

        img = cv.imread("predictshape.png")[:,:,0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])

        #predicting the images
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Custom Drawings Classifier",f"The drawing is probably a {self.class1}",parent=self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Custom Drawings Classifier",f"The drawing is probably a {self.class2}",parent=self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("Custom Drawings Classifier",f"The drawing is probably a {self.class3}",parent=self.root)

    ### Function to Rotate btw different Models ###
    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()

        #changes the status of the currently displayed model
        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    ### Function to Save the Current Model ###
    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path,"wb") as f:
            pickle.dump(self.clf,f)
        tkinter.messagebox.showinfo("Custom Drawings Classifier", "Model Successfully Saved", parent=self.root)

    ### Function to Load the Model ###
    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path,"rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo("Custom Drawings Classifier", "Model Successfully Loaded!", parent=self.root)

    ### Function to Save the Whole Project ###
    def save_everything(self):
        data = {"c1": self.class1, "c2": self.class2, "c3": self.class3, "c1c": self.class1_counter,
                "c2c": self.class2_counter, "c3c":self.class3_counter, "clf": self.clf, "pname": self.proj_name}
        #wb - writing bytes
        #rb - reading bytes
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle","wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Custom Drawings Classifier", "Model Successfully Saved!", parent=self.root)

    ### Function to Close the Application ###
    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?","Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

DrawingClassifier()