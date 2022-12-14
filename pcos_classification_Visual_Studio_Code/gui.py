from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import ImageTk,Image
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from os import listdir
import numpy as np
import pickle
import cv2
from os import listdir
import tensorflow as tf;
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import cv2
from imutils import paths
import os




EPOCHS =2
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((150, 150))
image_size = 150

width=256
height=256
depth=3


root = Tk()  # Main window 
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Polycystic Ovarian Disease Classification")
root.geometry("1080x720")
root.resizable(0,0)

canvas = Canvas(width=1080, height=250)
canvas.pack()
filename=('PCOD-in-adults.jpg')
load = Image.open(filename)
load = load.resize((1080, 250), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
load = Image.open(filename)
img.place(x=1, y=1)


root.configure(background='#FCFCE5')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)



classes = ['infected', 'notinfected']
len(classes)
from tensorflow.keras.models import load_model
model = load_model('./cnn.h5')

def clickbrowse():
	e2.delete("1.0","end-1c")
	global filename
	filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetypes =(("jpeg files","*.jpg"),("all files","*.*")) )

	load = Image.open(filename)
	load = load.resize((560,240), Image.ANTIALIAS)
	render = ImageTk.PhotoImage(load)
	img = Label(image=render)
	img.image = render
	img.place(x=250, y=340)
	
	imglist=[]
	image = cv2.imread(filename)
	image = cv2.resize(image, default_image_size)   
	img= img_to_array(image)
	
	
    
def click1():
	load = Image.open(filename)
	load = load.resize((560,240), Image.ANTIALIAS)
	render = ImageTk.PhotoImage(load)
	img = Label(image=render)
	img.image = render
	img.place(x=250, y=340)
	
	imglist=[]
	image = cv2.imread(filename)
	image = cv2.resize(image, default_image_size)   
	img= img_to_array(image)

	imglist.append(img)
	np_image_list = np.array(imglist)
	prediction=model.predict(np_image_list)
	
	classes_x=np.argmax(prediction,axis=1)
 
	id = classes_x
	if id == 0:
	 messagebox.showinfo('Prediction', 'infected')
	 e2.insert("1.0",'infected')
	elif id == 1:
	 messagebox.showinfo('Prediction', 'notinfected')
	 e2.insert("1.0",'notinfected')
	
	#e2.insert("1.0",classes[id])




def clear_all():  # for clearing the entry widgets
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()




label1 = Label(root, text="Polycystic Ovarian Disease Classification")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background="silver", fg="red", anchor="center")
label1.pack(fill=X)


frame2.pack_forget()
frame3.pack_forget()

e2 = Text(frame2, width=40,height=1)
e2.grid(row=1, column=2,padx=10)

e1 = Text(frame1,height=15, width=70)
e1.grid(row=1, column=2, padx=10,pady=10)


button5 = Button(frame3, text="Browse",command=clickbrowse,width=20,height=2)
button5.grid(row=12, column=1, pady=10,padx=10)

button5 = Button(frame3, text="Submit",command=click1,width=20,height=2)
button5.grid(row=13, column=1, pady=10,padx=10)


frame2.configure(background="silver")
frame2.pack(pady=10)

frame1.configure(background="red")
frame1.pack(pady=10)

frame3.configure(background="silver")
frame3.pack()

root.mainloop()
