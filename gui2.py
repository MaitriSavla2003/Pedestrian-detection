import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Loading the Model
model = load_model('Pedestrian_Detection2.h5')

# Modify the input shape of the model to (200, 200, 3)
new_input_shape = (200, 200, 3)
model.layers[0].batch_input_shape = (None,) + new_input_shape
model.build(input_shape=(None,) + new_input_shape)

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Pedestrian Detection')
top.configure(background='#CDCDCD')

# Initializing the Labels
label_result = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
sign_image = Label(top)

def detect_pedestrian(file_path):
    global label_result
    image = Image.open(file_path)
    image = image.resize((200, 200))  # Resize to match the new input shape
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255

    pred = model.predict(image)
    prediction = np.argmax(pred)

    classes = ["Pedestrian", "Not Pedestrian"]
    result_text = f"Predicted Class: {classes[prediction]}"

    label_result.configure(foreground="#011638", text=result_text)

def show_detect_button(file_path):
    detect_button = Button(top, text="Detect Pedestrian", command=lambda: detect_pedestrian(file_path), padx=10, pady=5)
    detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label_result.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(e)

upload_button = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload_button.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label_result.pack(side="bottom", expand=True)
heading = Label(top, text="Pedestrian Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
