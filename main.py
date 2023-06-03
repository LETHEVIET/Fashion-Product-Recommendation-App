import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image, ImageOps
import cv2

import pandas as pd
from pandas import DataFrame
import numpy as np

import io
import requests
import math

from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D

background_color = 'white'

dataset_path = './archive/images/'

width = 800
height = 600

K = 20
img_width, img_height, _ = 80, 60, 3 

def get_embedding(model, image):
    # img = keras.utils.load_img(path, target_size=(img_width, img_height))
    
    x   = keras.utils.img_to_array(image.resize((img_height, img_width)))
    x   = np.expand_dims(x, axis=0)
    x   = preprocess_input(x)
    return model.predict(x, verbose = 0).reshape(-1)

def recommendation_function(image):
  query_vector = get_embedding(CNN_model, image)
  _, indices = nn_model.kneighbors(np.expand_dims(query_vector, axis=0))
  return indices

def prepair_data():
    print('>> Preparing Data')
    global df, nn_model
    # df_embs = pd.read_csv("embeddings.csv")
    # df_embs.drop('Unnamed: 0', axis=1, inplace=True)
    df = pd.read_csv("metadata.csv")

    embeddings = np.load('embeddings.npy')
    nn_model = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(embeddings)

def resize_keep_ratio(image,  width, height):
    w, h = image.size

    wp = int(float(w * height) / h)
    hp = height

    if wp <= width and hp <=height:
        return image.resize((wp, hp))

    h = int(float(h * width) / w)
    w = width

    return image.resize((w, h))

def prepair_CNN_model():
    print('>> Preparing CNN model')
    global CNN_model
    # Input Shape
    img_width, img_height, _ = 80, 60, 3 #load_image(df.iloc[0].image).shape

    # Pre-Trained Model
    base_model = ResNet50(weights='imagenet', 
                        include_top=False, 
                        input_shape = (img_width, img_height, 3))
    base_model.trainable = False

    # Add Layer Embedding
    CNN_model = keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

def get_url_image(url):
  image = Image.open(requests.get(url, stream = True).raw)
  return image#.resize((120,160))

def detail_2_list():
    frm_products_Detail.grid_remove()
    frm_products_container.grid(row=1, column=0, sticky='nwse')

def on_product_clicked(index):
    print(index)

    for child in frm_products_Detail.winfo_children():
        child.destroy()

    data = df.iloc[index]
    print(data.link)
    url_image = get_url_image(data.link)

    url_image = resize_keep_ratio(url_image,380, 300)  # Adjust the size as needed
    photo = ImageTk.PhotoImage(url_image)

    lbl_product_image = tk.Label(frm_products_Detail, image=photo, width=380, height=300)
    lbl_product_image.image = photo
    lbl_product_image.grid(row=0, column=0, columnspan=2, sticky="nsew")
    lbl_product_name = tk.Label(frm_products_Detail, text=data.productDisplayName)
    lbl_product_name.grid(row=1, column=0, columnspan=2, sticky="nsew")

    for i, field in enumerate(['gender','masterCategory','subCategory','articleType','baseColour','season','year']):
        lbl_product_details = tk.Label(frm_products_Detail, text=f'{field}')
        lbl_product_details.grid(row=2 + i, column=0, sticky="nsew")

        lbl_product_details2 = tk.Label(frm_products_Detail, text=f'{data[field]}')
        lbl_product_details2.grid(row=2 + i, column=1, sticky="nsew")

    btn_detail_2_list = tk.Button(frm_products_Detail, text="Back", command=detail_2_list)
    btn_detail_2_list.grid(row=9, column=0, columnspan=2,sticky="nsew")

    frm_products_container.grid_remove()
    frm_products_Detail.grid(row=1, column=0, sticky='nwse')

def predict():
    global query_image
    indices = recommendation_function(query_image)
    print(indices)
    products = [{"text": row.productDisplayName, 
                "image": Image.open(dataset_path + row.image), 
                "index": i} for i, row in df.loc[indices[0], :].iterrows()]
    
    number_product_per_row = 2
    for i in range(number_product_per_row):
        frm_product_inner_container.columnconfigure(i, weight=1)

    for child in frm_product_inner_container.winfo_children():
        child.destroy()

    for i in range(int(math.ceil(K/number_product_per_row))):
        start = i * number_product_per_row
        end = min(start + number_product_per_row - 1, K)
        # print('..',(start, end))
        for index in range(start, end+1):
            frame = tk.Frame(
                master=frm_product_inner_container,
                relief=tk.RAISED,
                # bg=background_color,
                width=196,
                height=150,
            )
            frame.pack_propagate(False)

            col = 0 if index == 0 else index%number_product_per_row
            frame.grid(row=i, column=col, pady=2, padx=2, sticky='we')

            # Load the image and create a PhotoImage object
            image = products[index]["image"]
            # image = image.resize((60, 80))  # Adjust the size as needed
            photo = ImageTk.PhotoImage(image)
            
            # Create the label and display the image
            label_image = tk.Label(master=frame, image=photo, bg=background_color)
            label_image.image = photo  # Save a reference to prevent garbage collection
            label_image.pack(padx=5, pady=5)
           
            label_text = tk.Label(master=frame, text=products[index]["text"], bg=background_color, wraplength=150)
            label_text.pack(padx=5, pady=5)   
            
            frame.bind("<Button-1>", lambda event, index=products[index]["index"]: on_product_clicked(index))     
            label_image.bind("<Button-1>", lambda event, index=products[index]["index"]: on_product_clicked(index))     
            label_text.bind("<Button-1>", lambda event, index=products[index]["index"]: on_product_clicked(index))     

            window.update()
            window.update_idletasks()
            update_scroll_region('_')
            
def stop_capturing():
    cav_capture.after_cancel(capture_after_id)
    cap.release()

def upload_image():
    global file_path

    frm_main_left.grid_remove()
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files','*.png')]
    file_path = filedialog.askopenfilename(filetypes=f_types)

    lbl_main_left_title.config(text=file_path)

    print("File was choosed: ", file_path)
    
    global new_pic, query_image
    query_image = Image.open(file_path)
    cav_crop.update()
    print(cav_crop.winfo_width(), cav_crop.winfo_height())
    image6b = resize_keep_ratio(query_image, cav_crop.winfo_width(), cav_crop.winfo_height())
    print(image6b.size)

    new_pic = ImageTk.PhotoImage(image6b)
    pos_x = (cav_crop.winfo_width() - new_pic.width()) / 2
    pos_y = (cav_crop.winfo_height() - new_pic.height()) / 2

    cav_crop.coords(image_container, pos_x, pos_y)
    cav_crop.itemconfig(image_container,image=new_pic)

    frm_crop_image.grid(row=1, column=0, sticky= "wesn")

def crop_2_main_left():
    lbl_main_left_title.config(text='Input Selections')
    frm_crop_image.grid_remove()
    frm_main_left.grid(row=1, column=0, sticky= "wesn")

def capture_2_main_left():
    lbl_main_left_title.config(text='Input Selections')
    stop_capturing()
    frm_capture.grid_remove()
    frm_main_left.grid(row=1, column=0, sticky= "wesn")

def snap_picture():
    stop_capturing()
    frm_capture.grid_remove()
    global new_pic
    pos_x = (cav_crop.winfo_width() - new_pic.width()) / 2
    pos_y = (cav_crop.winfo_height() - new_pic.height()) / 2

    cav_crop.coords(image_container, pos_x, pos_y)
    cav_crop.itemconfig(image_container,image=new_pic)

    frm_crop_image.grid(row=1, column=0, sticky= "wesn")

def update_frame():
    
    global new_pic, capture_after_id, query_image
    window.update()
    ret, frame = cap.read()  # Read a frame from the webcam
    if ret:
        # Convert the frame to PIL Image format
        query_image = ImageOps.mirror(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        image6b = resize_keep_ratio(query_image, cav_capture.winfo_width(), cav_capture.winfo_height())

        new_pic = ImageTk.PhotoImage(image6b)
        pos_x = (cav_capture.winfo_width() - new_pic.width()) // 2
        pos_y = (cav_capture.winfo_height() - new_pic.height()) // 2

        cav_capture.coords(capture_container, pos_x, pos_y)
        cav_capture.itemconfig(capture_container,image=new_pic)
        
    # Schedule the next update after a delay
    capture_after_id = cav_capture.after(10, update_frame)

def open_camera():
    lbl_main_left_title.config(text='Captured Image')

    frm_main_left.grid_remove()
    frm_capture.grid(row=1, column=0, sticky= "wesn")
    # Open the webcam
    global cap
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cav_capture.winfo_width())
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cav_capture.winfo_height())
    # Start updating the video stream
    update_frame()

window = tk.Tk()
window.geometry(f'{width}x{height}')

window.minsize(width, height)
window.maxsize(width, height)

window.title('Fashion Product Recommendation System')
window.configure(background='white')

prepair_CNN_model()
prepair_data()

window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)

frm_top = tk.Frame(master=window, borderwidth=1, height=50, bg='red')
frm_content = tk.Frame(master=window, borderwidth=1, bg=background_color)
frm_top.grid(row=0, column=0, sticky='ew')
frm_content.grid(row=1, column=0, sticky='nsew')
frm_content.grid_propagate(False)

frm_content.columnconfigure(0, weight=1)
frm_content.columnconfigure(1, weight=1)
frm_content.rowconfigure(0, weight=1)

frm_left = tk.Frame(master=frm_content, borderwidth=1)
frm_right = tk.Frame(master=frm_content, borderwidth=1)

frm_left.grid(row=0, column=0, sticky='nsew')
frm_left.grid_propagate(False)
frm_right.grid(row=0, column=1, sticky='nsew')
frm_right.grid_propagate(False)

# Left Frame
frm_left.columnconfigure(0, weight=1)
frm_left.rowconfigure(1, weight=1)

lbl_main_left_title = tk.Label(master=frm_left, text='Input Selections')
lbl_main_left_title.grid(row=0, column=0, sticky='nwse')

## Main Left Frame

frm_main_left = tk.Frame(master=frm_left, bg=background_color)
frm_main_left.grid(row=1, column=0, sticky= "wesn")

frm_main_left.columnconfigure(0, weight=1)
frm_main_left.rowconfigure(0, weight=1)

frm_main_button_group = tk.Frame(master=frm_main_left, bg=background_color)
frm_main_button_group.grid(row=0, column=0)

btn_choose_image = tk.Button(master=frm_main_button_group, text="Choose Image", command=upload_image)
btn_capture_image = tk.Button(master=frm_main_button_group, text="Capture Image", command=open_camera)
btn_choose_image.pack(pady=5)
btn_capture_image.pack(pady=5)


## Crop Image Frame

frm_crop_image = tk.Frame(master=frm_left, bg=background_color)
frm_crop_image.grid(row=1, column=0, sticky= "wesn")
frm_crop_image.columnconfigure(0, weight=1)
frm_crop_image.rowconfigure(1, weight=1)

cav_crop = tk.Canvas(master=frm_crop_image, height = 350, background=background_color)
cav_crop.grid(row=0, column=0, sticky='wen')
cav_crop.update()

frm_crop_button_group = tk.Frame(master=frm_crop_image)
frm_crop_button_group.grid(row=1, column=0, sticky='esnw')

btn_crop_2_main_left = tk.Button(master=frm_crop_button_group, text="Back", command=crop_2_main_left, padx=5)
btn_crop_2_main_left.place(relx=0.0, rely=0.5, anchor='w')
btn_crop_predict = tk.Button(master=frm_crop_button_group, text="Predict", command=predict, padx=5)
btn_crop_predict.place(relx=0.5, rely=0.5, anchor='center')

uploaded = Image.open('black.jpg')
image6b = resize_keep_ratio(uploaded, cav_crop.winfo_width(), cav_crop.winfo_height())
python_image6b = ImageTk.PhotoImage(image6b)

pos_x = (cav_crop.winfo_width() - image6b.size[0]) // 2
pos_y = (cav_crop.winfo_height() - image6b.size[1]) // 2
image_container = cav_crop.create_image(pos_x,pos_y,anchor=NW,image=python_image6b)
frm_crop_image.grid_remove()

## capture frame

frm_capture = tk.Frame(master=frm_left, bg=background_color)
frm_capture.grid(row=1, column=0, sticky= "wesn")
frm_capture.grid_propagate(False)
frm_capture.columnconfigure(0, weight=1)
frm_capture.rowconfigure(1, weight=1)

cav_capture = tk.Canvas(master=frm_capture, height = 350, background=background_color)
cav_capture.grid(row=0, column=0, sticky='enw')
cav_capture.update()

frm_capture_button_group = tk.Frame(master=frm_capture)
frm_capture_button_group.grid(row=1, column=0, sticky='esnw')

btn_capture_2_main_left = tk.Button(master=frm_capture_button_group, text="Back", command=capture_2_main_left, padx=5)
btn_capture_2_main_left.place(relx=0.0, rely=0.5, anchor='w')
btn_capture_snap = tk.Button(master=frm_capture_button_group, text="Snap", command=snap_picture, padx=5)
btn_capture_snap.place(relx=0.5, rely=0.5, anchor='center')


uploaded = Image.open('black.jpg')
image6b = resize_keep_ratio(uploaded, cav_capture.winfo_width(), cav_capture.winfo_height())
python_image6b = ImageTk.PhotoImage(image6b)

pos_x = (cav_capture.winfo_width() - image6b.size[0]) / 2
pos_y = (cav_capture.winfo_height() - image6b.size[1]) / 2
capture_container = cav_capture.create_image(pos_x,pos_y,anchor=NW,image=python_image6b)
frm_capture.grid_remove()

# Right Frame

frm_right.columnconfigure(0, weight=1)
frm_right.rowconfigure(1, weight=1)

## Main Right Frame

lbl_main_right_title = tk.Label(master=frm_right, text='Recommendation products')
lbl_main_right_title.grid(row=0, column=0, sticky='nwse')
## Products Container
frm_products_container = tk.Frame(master=frm_right, bg=background_color)
frm_products_container.grid(row=1, column=0, sticky='nwse')
frm_products_container.columnconfigure(0, weight=1)
frm_products_container.rowconfigure(0, weight=1)

cav_product_container = tk.Canvas(frm_products_container, bg=background_color)
cav_product_container.grid(row=0, column=0, sticky="nsew")

# Create a scrollbar
scrollbar = tk.Scrollbar(frm_products_container, orient=tk.VERTICAL, command=cav_product_container.yview)
scrollbar.grid(row=0, column=1, sticky="ns")

# Configure the cav_product_container to use the scrollbar
cav_product_container.configure(yscrollcommand=scrollbar.set)

cav_product_container.update()
frm_product_inner_container = tk.Frame(cav_product_container, width=380, bg=background_color)
frm_product_inner_container.update()
# frm_product_inner_container.grid_propagate(False)
# Add widgets to the inner frame

cav_product_container.create_window((0, 0), window=frm_product_inner_container, anchor=tk.NW, width=cav_product_container.winfo_width())

def update_scroll_region(event):
    cav_product_container.configure(scrollregion=cav_product_container.bbox("all"))

cav_product_container.bind('<Configure>', update_scroll_region)
# Update the display and process events
window.update()
window.update_idletasks()

# Manually trigger the canvas configuration
update_scroll_region('_')

## Product Detail
frm_products_Detail = tk.Frame(master=frm_right)
# frm_products_Detail.grid(row=1, column=0, sticky='nwse')
frm_products_Detail.grid_propagate(False)
# frm_products_Detail.columnconfigure(0, weight=1)
# frm_products_Detail.rowconfigure(0, weight=1)

print('>> Program Started')
window.mainloop()
