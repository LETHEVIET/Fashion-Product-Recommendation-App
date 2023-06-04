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


WIDTH = 800
HEIGHT = 600

K = 20

img_width, img_height, _ = 80, 60, 3 

shopee_color = '#fb5630'

LARGEFONT =("Verdana", 35)

dataset_path = "./archive"

class Tkinter_app(tk.Tk):
    
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.prepair_data()
        self.prepair_CNN_model()

        self.geometry(f'{WIDTH}x{HEIGHT}')

        self.minsize(WIDTH, HEIGHT)
        self.maxsize(WIDTH, HEIGHT)

        self.title('Shopee')
        self.configure(background='white')

        # creating a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        # initializing frames to an empty array
        self.frames = {} 

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (Home_page,Recommendation_page):

            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row = 0, column = 0, sticky ="nsew")

        self.show_frame(Home_page)


    # to display the current frame passed as
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    
    def prepair_data(self):
        print('>> Preparing Data')
        global df, nn_model
        df = pd.read_csv("metadata.csv")

        embeddings = np.load('embeddings.npy')
        nn_model = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute', n_jobs=-1)
        nn_model.fit(embeddings)

    
    def prepair_CNN_model(self):
        print('>> Preparing CNN model')
        global CNN_model
        
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

class Home_page(tk.Frame):
    def __init__(self, parent, controller):
        global df
        tk.Frame.__init__(self, parent)
                
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_columnconfigure(0, weight = 1)

        frm_topbar = tk.Frame(container, bg=shopee_color, height=60, padx=10)
        frm_topbar.grid(row=0, column=0, sticky='nwse')
        frm_topbar.rowconfigure(0, weight=1)
        frm_topbar.columnconfigure(1, weight=1)
        frm_topbar.grid_propagate(False)

        img_shopeeicon = ImageTk.PhotoImage(self.resize_keep_ratio(Image.open('imgs/shopee_icon.png'), 80, 80))
        lbl_shoppeicon = tk.Label(frm_topbar, bg=shopee_color, image=img_shopeeicon)
        lbl_shoppeicon.image = img_shopeeicon
        lbl_shoppeicon.grid(row=0,column=0, sticky='nwse')

        frm_searchbar = tk.Frame(frm_topbar, bg=shopee_color)
        frm_searchbar.grid(row=0, column=1, sticky='nwse', pady=10, padx=10)
        frm_searchbar.rowconfigure(0, weight=1)
        frm_searchbar.columnconfigure(0, weight=1)
        frm_searchbar.grid_propagate(False)

        lbl_searchbar = tk.Label(frm_searchbar, bg='white', text='Ôn lại bí kíp săn sale', fg='#ccc', anchor='w', padx=5)
        lbl_searchbar.grid(row=0, column=0, sticky='nwse')

        frm_searchbtn = tk.Frame(frm_searchbar, bg='white', padx=2, pady=2, width=50)
        frm_searchbtn.grid(row=0, column=1, sticky='nwse')
        frm_searchbtn.grid_propagate(False)
        frm_searchbtn.rowconfigure(0, weight=1)
        frm_searchbtn.columnconfigure(0, weight=1)

        img_searchicon = ImageTk.PhotoImage(Image.open('imgs/search_icon.png'))
        lbl_searchbtn = tk.Label(frm_searchbtn, bg=shopee_color, image=img_searchicon, padx=3, pady=3)
        lbl_searchbtn.image = img_searchicon
        lbl_searchbtn.grid(row=0, column=0, sticky='nwse')

        btn_imgrecommend = tk.Button(frm_topbar, bg='#D0011B', fg='white', text='GỢI Ý BẰNG HÌNH ẢNH'
                                    ,borderwidth=0, relief="solid", padx=10, font=('Arial', 9), 
                                     command=lambda : controller.show_frame(Recommendation_page))
        btn_imgrecommend.grid(row=0, column=3, sticky='nwse', pady=10, padx=10)

        banner_width = 795
        banner1_height = int(banner_width/(3+3/2))
        frm_banner = tk.Frame(container, bg='black')
        frm_banner.grid(row=2, column=0, sticky='nwse')
        
        img_banner1 = ImageTk.PhotoImage(Image.open('imgs/Banner AI.png').resize((banner1_height*3,banner1_height)))
        lbl_banner1 = tk.Label(frm_banner, image=img_banner1)
        lbl_banner1.image = img_banner1
        lbl_banner1.grid(row=0, column=0, rowspan=2, sticky='nwse')
        
        img_banner2 = ImageTk.PhotoImage(Image.open('imgs/banner2.jpg').resize((int(banner1_height/2)*3,int(banner1_height/2))))
        lbl_banner2 = tk.Label(frm_banner, image=img_banner2)
        lbl_banner2.image = img_banner2
        lbl_banner2.grid(row=0, column=1, sticky='nwse')

        img_banner3 = ImageTk.PhotoImage(Image.open('imgs/banner3.jpg').resize((int(banner1_height/2)*3,int(banner1_height/2))))
        lbl_banner3 = tk.Label(frm_banner, image=img_banner3)
        lbl_banner3.image = img_banner3
        lbl_banner3.grid(row=1, column=1, sticky='nwse')
        
        frm_boxes = tk.Frame(container, bg='white')
        frm_boxes.grid(row=3, column=0, sticky='nwse')

        boxes_img_list = ["imgs/b1.png","imgs/b2.png","imgs/b3.png","imgs/b4.png","imgs/b5.png","imgs/b6.png","imgs/b7.png"]
        boxes_txt_list = ["Khung Giờ Săn Sale", "Miễn Phí Vận Chuyển", "Voucher Giảm Đến 200.000Đ", "Hàng Hiệu Outlet Giảm 50%", "Mã Giảm Giá", "Bắt Trend - Giá Sốc", "Nạp Thẻ, Dịch Vụ & Data"]
        for i, img_path in enumerate(boxes_img_list[:6]):
            frm_boxes.columnconfigure(i, weight=1)
            img_box = ImageTk.PhotoImage(Image.open(img_path).resize((50,50)))
            lbl_box = tk.Label(frm_boxes, image=img_box, bg='white')
            lbl_box.image = img_box
            lbl_box.grid(row=0, column=i, sticky='nwse')
            lbl_box_text = tk.Label(frm_boxes, text=boxes_txt_list[i], bg='white', wraplength=80)
            lbl_box_text.grid(row=1, column=i, sticky='nwse')

        frm_foryou = tk.Frame(container, bg=shopee_color)
        frm_foryou.grid(row=4, column=0, sticky='nwse',pady=10)
        frm_foryou.columnconfigure(0, weight=1)
        
        lbl_foryou = tk.Label(frm_foryou, text="GỢI Ý HÔM NAY", bg='white', fg=shopee_color)
        lbl_foryou.grid(row=0, column=0, sticky='nwse')

        frm_foryou_products = tk.Frame(frm_foryou, bg=shopee_color) 
        frm_foryou_products.grid(row=1, column=0, sticky='nwse')
        foryou_list =  [row for i, row in df.sample(8).iterrows()]
        for i, product in enumerate(foryou_list[:4]):
            frm_foryou_products.columnconfigure(i, weight=1)
            frm_box = tk.Frame(frm_foryou_products, bg='white')
            frm_box.grid(row=0,column=i, sticky='nwse', padx=5, pady=5)
            frm_box.columnconfigure(i, weight=1)
            frm_box.rowconfigure(1, weight=1)
            img_box = ImageTk.PhotoImage(Image.open(dataset_path + '/images/' + product.image))
            lbl_box = tk.Label(frm_box, image=img_box, bg='white')
            lbl_box.image = img_box
            lbl_box.grid(row=0, column=i, sticky='nwse')
            lbl_box_text = tk.Label(frm_box, text=product.productDisplayName, wraplength=150)
            lbl_box_text.grid(row=1, column=i, sticky='nwse')

        for i, product in enumerate(foryou_list[4:]):
            frm_foryou_products.columnconfigure(i, weight=1)
            frm_box = tk.Frame(frm_foryou_products, bg='white')
            frm_box.grid(row=1,column=i, sticky='nwse', padx=5, pady=5)
            frm_box.columnconfigure(i, weight=1)
            frm_box.rowconfigure(1, weight=1)
            img_box = ImageTk.PhotoImage(Image.open(dataset_path + '/images/' + product.image))
            lbl_box = tk.Label(frm_box, image=img_box, bg='white')
            lbl_box.image = img_box
            lbl_box.grid(row=0, column=i, sticky='nwse')
            lbl_box_text = tk.Label(frm_box, text=product.productDisplayName, wraplength=150)
            lbl_box_text.grid(row=1, column=i, sticky='nwse')
    

    def resize_keep_ratio(self, image,  width, height):
        w, h = image.size

        wp = int(float(w * height) / h)
        hp = height

        if wp <= width and hp <=height:
            return image.resize((wp, hp))

        h = int(float(h * width) / w)
        w = width

        return image.resize((w, h))

class Recommendation_page(tk.Frame):
    def __init__(self, parent, controller):
        global df

        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.products = []
        self.products_widget = []   
                
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(1, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        frm_topbar = tk.Frame(container, bg=shopee_color, height=60, padx=10)
        frm_topbar.grid(row=0, column=0, sticky='nwse')
        frm_topbar.rowconfigure(0, weight=1)
        frm_topbar.columnconfigure(1, weight=1)
        frm_topbar.grid_propagate(False)

        img_shopeeicon = ImageTk.PhotoImage(self.resize_keep_ratio(Image.open('imgs/shopee_icon.png'), 80, 80))

        lbl_shoppeicon = tk.Label(frm_topbar, bg=shopee_color, image=img_shopeeicon)
        lbl_shoppeicon.image = img_shopeeicon
        lbl_shoppeicon.grid(row=0,column=0, sticky='nwse')

        frm_searchbar = tk.Frame(frm_topbar, bg=shopee_color)
        frm_searchbar.grid(row=0, column=1, sticky='nwse', pady=10, padx=10)
        frm_searchbar.rowconfigure(0, weight=1)
        frm_searchbar.columnconfigure(0, weight=1)
        frm_searchbar.grid_propagate(False)

        lbl_searchbar = tk.Label(frm_searchbar, bg='white', text='Ôn lại bí kíp săn sale', fg='#ccc', anchor='w', padx=5)
        lbl_searchbar.grid(row=0, column=0, sticky='nwse')

        frm_searchbtn = tk.Frame(frm_searchbar, bg='white', padx=2, pady=2, width=50)
        frm_searchbtn.grid(row=0, column=1, sticky='nwse')
        frm_searchbtn.grid_propagate(False)
        frm_searchbtn.rowconfigure(0, weight=1)
        frm_searchbtn.columnconfigure(0, weight=1)

        img_searchicon = ImageTk.PhotoImage(Image.open('imgs/search_icon.png'))
        lbl_searchbtn = tk.Label(frm_searchbtn, bg=shopee_color, image=img_searchicon, padx=3, pady=3)
        lbl_searchbtn.image = img_searchicon
        lbl_searchbtn.grid(row=0, column=0, sticky='nwse')

        btn_imgrecommend = tk.Button(frm_topbar, bg='#D0011B', fg='white', text='Trở về trang chủ'
                                    ,borderwidth=0, relief="solid", padx=10, font=('Arial', 9),
                                     command=lambda : controller.show_frame(Home_page))
        btn_imgrecommend.grid(row=0, column=3, sticky='nwse', pady=10, padx=10)

        frm_content = tk.Frame(container, bg='white')
        frm_content.grid(row=1, column=0, sticky='nsew')
        frm_content.grid_propagate(False)
        frm_content.columnconfigure(0, weight=1)
        frm_content.columnconfigure(1, weight=1)
        frm_content.rowconfigure(0, weight=1)

        frm_left = tk.Frame(master=frm_content, bg='white')
        frm_right = tk.Frame(master=frm_content, bg=shopee_color)

        frm_left.grid(row=0, column=0, sticky='nsew')
        frm_left.grid_propagate(False)
        frm_right.grid(row=0, column=1, sticky='nsew')
        frm_right.grid_propagate(False)

        # Left Frame
        frm_left.columnconfigure(0, weight=1)
        frm_left.rowconfigure(1, weight=1)

        lbl_main_left_title = tk.Label(master=frm_left, text='Chọn Phương Thức Chọn Ảnh',
                                        bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                        pady=10, font=('Arial', 9))
        self.lbl_main_left_title = lbl_main_left_title
        lbl_main_left_title.grid(row=0, column=0, sticky='nwse')

        ## Main Left Frame

        frm_main_left = tk.Frame(master=frm_left, bg='white')
        self.frm_main_left = frm_main_left
        frm_main_left.grid(row=1, column=0, sticky= "wesn")

        frm_main_left.columnconfigure(0, weight=1)
        frm_main_left.rowconfigure(0, weight=1)

        frm_main_button_group = tk.Frame(master=frm_main_left, bg='white')
        frm_main_button_group.grid(row=0, column=0)

        btn_choose_image = tk.Button(master=frm_main_button_group, text="Tải Hình", command=self.upload_image,
                                    bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                    width=8, padx=10, pady=5, font=('Arial', 9))
        btn_capture_image = tk.Button(master=frm_main_button_group, text="Chụp Hình", command=self.open_camera,
                                      bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                      width=8, padx=10, pady=5, font=('Arial', 9))
        btn_choose_image.pack(pady=5)
        btn_capture_image.pack(pady=5)
        
        ## Crop Image Frame

        frm_view_image = tk.Frame(master=frm_left)
        self.frm_view_image = frm_view_image
        frm_view_image.grid(row=1, column=0, sticky= "wesn")
        frm_view_image.columnconfigure(0, weight=1)
        frm_view_image.rowconfigure(1, weight=1)

        cav_view = tk.Canvas(master=frm_view_image, height = 350, width=400, bg='black')
        self.cav_view = cav_view
        cav_view.grid(row=0, column=0, sticky='wen')
        cav_view.update()

        frm_view_button_group = tk.Frame(master=frm_view_image, bg='white', padx=10)
        frm_view_button_group.grid(row=1, column=0, sticky='esnw')

        btn_view_2_main_left = tk.Button(master=frm_view_button_group, text="Trở về", command=self.crop_2_main_left, 
                                         bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                         width=8, padx=10, pady=5, font=('Arial', 9))
        btn_view_2_main_left.place(relx=0.0, rely=0.5, anchor='w')
        btn_view_predict = tk.Button(master=frm_view_button_group, text="Gợi ý", command=self.predict, 
                                     bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                     width=8, padx=10, pady=5, font=('Arial', 9))

        btn_view_predict.place(relx=0.5, rely=0.5, anchor='center')

        uploaded = Image.open('imgs/black.jpg')
        image6b = self.resize_keep_ratio(uploaded, 400, cav_view.winfo_height())
        python_image6b = ImageTk.PhotoImage(image6b)
        self.python_image6b = python_image6b

        global image_container
        pos_x = (400 - image6b.size[0]) // 2
        pos_y = (cav_view.winfo_height() - image6b.size[1]) // 2
        self.image_container = cav_view.create_image(pos_x,pos_y,anchor=NW,image=python_image6b)

        frm_view_image.grid_remove()

        ## capture frame

        frm_capture = tk.Frame(frm_left, bg='white')
        self.frm_capture = frm_capture
        frm_capture.grid(row=1, column=0, sticky= "wesn")
        frm_capture.grid_propagate(False)
        frm_capture.columnconfigure(0, weight=1)
        frm_capture.rowconfigure(1, weight=1)

        cav_capture = tk.Canvas(master=frm_capture, height = 350, background='white')
        self.cav_capture = cav_capture
        cav_capture.grid(row=0, column=0, sticky='enw')
        cav_capture.update()

        frm_capture_button_group = tk.Frame(master=frm_capture, padx=10)
        frm_capture_button_group.grid(row=1, column=0, sticky='esnw')

        btn_capture_2_main_left = tk.Button(master=frm_capture_button_group, text="Trở về", command=self.capture_2_main_left,
                                            bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                            width=8, padx=10, pady=5, font=('Arial', 9))
        btn_capture_2_main_left.place(relx=0.0, rely=0.5, anchor='w')
        btn_capture_snap = tk.Button(master=frm_capture_button_group, text="Chụp ảnh", command=self.snap_picture, 
                                     bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                     width=8, padx=10, pady=5, font=('Arial', 9))

        btn_capture_snap.place(relx=0.5, rely=0.5, anchor='center')

        pos_x = (cav_capture.winfo_width() - image6b.size[0]) / 2
        pos_y = (cav_capture.winfo_height() - image6b.size[1]) / 2
        capture_container = cav_capture.create_image(pos_x,pos_y,anchor=NW,image=python_image6b)
        self.capture_container = capture_container
        frm_capture.grid_remove()

        # Right Frame

        frm_right.columnconfigure(0, weight=1)
        frm_right.rowconfigure(1, weight=1)

        ## Main Right Frame

        lbl_main_right_title = tk.Label(master=frm_right, text='Sản Phẩm Gợi Ý',
                                        bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                        pady=10, font=('Arial', 9))

        self.lbl_main_right_title = lbl_main_right_title
        lbl_main_right_title.grid(row=0, column=0, sticky='nwse')
        
        ## Products Container
        
        frm_products_container = tk.Frame(master=frm_right, bg='white')
        self.frm_products_container = frm_products_container
        frm_products_container.grid(row=1, column=0, sticky='nwse')
        frm_products_container.columnconfigure(0, weight=1)
        frm_products_container.rowconfigure(0, weight=1)

        vscrollbar = tk.Scrollbar(frm_products_container, orient=VERTICAL)
        vscrollbar.grid(row=0, column=1, sticky="ns")
        canvas = tk.Canvas(frm_products_container, bd=0, highlightthickness=0,
                           yscrollcommand=vscrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.frm_product_inner_container = frm_product_inner_container = tk.Frame(canvas, width=380, bg=shopee_color, pady=2, padx=2)
        frm_product_inner_container_id = canvas.create_window(0, 0, window=frm_product_inner_container, anchor=NW)

        def _configure_frm_product_inner_container(event):
            size = (frm_product_inner_container.winfo_reqwidth(), frm_product_inner_container.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if frm_product_inner_container.winfo_reqwidth() != canvas.winfo_width():
                canvas.config(width=frm_product_inner_container.winfo_reqwidth())
        frm_product_inner_container.bind('<Configure>', _configure_frm_product_inner_container)

        def _configure_canvas(event):
            if frm_product_inner_container.winfo_reqwidth() != canvas.winfo_width():
                canvas.itemconfigure(frm_product_inner_container_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

        ## Product Detail
        frm_products_Detail = tk.Frame(master=frm_right)
        self.frm_products_Detail = frm_products_Detail
        frm_products_Detail.grid_propagate(False)

    def get_url_image(self,url):
        image = Image.open(requests.get(url, stream = True).raw)
        return image

    def detail_2_list(self):
        self.frm_products_Detail.grid_remove()
        self.frm_products_container.grid(row=1, column=0, sticky='nwse')

    def on_product_clicked(self,index):
        print(index)

        for child in self.frm_products_Detail.winfo_children():
            child.destroy()

        data = df.iloc[index]
        print(data.link)
        url_image = self.get_url_image(data.link)

        url_image = self.resize_keep_ratio(url_image,380, 250)  # Adjust the size as needed
        photo = ImageTk.PhotoImage(url_image)

        lbl_product_image = tk.Label(self.frm_products_Detail, image=photo, bg=shopee_color, width=380, height=250)
        lbl_product_image.image = photo
        lbl_product_image.grid(row=0, column=0, columnspan=2, sticky="nsew")
        lbl_product_name = tk.Label(self.frm_products_Detail, font=('Arial', 12, 'bold'), text=data.productDisplayName)
        lbl_product_name.grid(row=1, column=0, columnspan=2, sticky="nsew")

        for i, field in enumerate(['gender','masterCategory','subCategory','articleType','baseColour','season','year']):
            lbl_product_details = tk.Label(self.frm_products_Detail, text=f'{field}')
            lbl_product_details.grid(row=2 + i, column=0, sticky="nsew")

            lbl_product_details2 = tk.Label(self.frm_products_Detail, text=f'{data[field]}')
            lbl_product_details2.grid(row=2 + i, column=1, sticky="nsew")

        btn_detail_2_list = tk.Button(self.frm_products_Detail, text="Back", command=self.detail_2_list,
                                     bg='#D0011B', fg='white', borderwidth=0, relief="solid", 
                                     width=8, padx=10, pady=5, font=('Arial', 9))
        btn_detail_2_list.grid(row=9, column=0, columnspan=2, pady=5)

        self.frm_products_container.grid_remove()
        self.frm_products_Detail.grid(row=1, column=0, sticky='nwse')

    def predict(self):
        global query_image
        indices = self.recommendation_function(query_image)
        print(indices)
        self.products = [{"text": row.productDisplayName, 
                          "image": ImageTk.PhotoImage(Image.open(dataset_path + '/images/' + row.image)), 
                          "index": i} for i, row in df.loc[indices[0], :].iterrows()]   
        number_product_per_row = 2
        for i in range(number_product_per_row):
            self.frm_product_inner_container.columnconfigure(i, weight=1)

        for child in self.frm_product_inner_container.winfo_children():
            child.destroy()

        for i in range(int(math.ceil(K/number_product_per_row))):
            start = i * number_product_per_row
            end = min(start + number_product_per_row - 1, K)
            # print('..',(start, end))
            for index in range(start, end+1):
                frame = tk.Frame(
                    master=self.frm_product_inner_container,
                    relief=tk.RAISED,
                    width=196,
                    height=150,
                )
                frame.pack_propagate(False)

                col = 0 if index == 0 else index%number_product_per_row
                frame.grid(row=i, column=col, pady=2, padx=2, sticky='wesn')
                frame.columnconfigure(i, weight=1)
                frame.rowconfigure(1, weight=1)

                # Create the label and display the image
                label_image = tk.Label(master=frame, image=self.products[index]["image"], bg='white')
                label_image.image = self.products[index]["image"]  # Save a reference to prevent garbage collection
                label_image.grid(row=0, column=i, sticky='nwse')
            
                label_text = tk.Label(master=frame, text=self.products[index]["text"], fg='white', bg='#D0011B', wraplength=150)
                label_text.grid(row=1, column=i, sticky='nwse')
                
                frame.bind("<Button-1>", lambda event, index=self.products[index]["index"]: self.on_product_clicked(index))     
                label_image.bind("<Button-1>", lambda event, index=self.products[index]["index"]: self.on_product_clicked(index))     
                label_text.bind("<Button-1>", lambda event, index=self.products[index]["index"]: self.on_product_clicked(index))     

                self.parent.update()
                self.parent.update_idletasks()
                
    def stop_capturing(self):
        self.cav_capture.after_cancel(capture_after_id)
        cap.release()

    def upload_image(self):
        global file_path

        self.frm_main_left.grid_remove()
        f_types = [('Jpg Files', '*.jpg'), ('PNG Files','*.png')]
        file_path = filedialog.askopenfilename(filetypes=f_types)

        self.lbl_main_left_title.config(text=file_path)

        print("File was choosed: ", file_path)
        
        global new_pic, query_image
        query_image = Image.open(file_path)
        self.cav_view.update()
        
        image6b = self.resize_keep_ratio(query_image, 400, self.cav_view.winfo_height())
        print(image6b.size)

        new_pic = ImageTk.PhotoImage(image6b)
        pos_x = (400 - new_pic.width()) / 2
        pos_y = (self.cav_view.winfo_height() - new_pic.height()) / 2

        self.cav_view.coords(self.image_container, pos_x, pos_y)
        self.cav_view.itemconfig(self.image_container,image=new_pic)

        self.frm_view_image.grid(row=1, column=0, sticky= "wesn")

    def crop_2_main_left(self):
        self.lbl_main_left_title.config(text='Chọn Phương Thức Chọn Ảnh')
        self.frm_view_image.grid_remove()
        self.frm_main_left.grid(row=1, column=0, sticky= "wesn")

    def capture_2_main_left(self):
        self.lbl_main_left_title.config(text='Chọn Phương Thức Chọn Ảnh')
        self.stop_capturing()
        self.frm_capture.grid_remove()
        self.frm_main_left.grid(row=1, column=0, sticky= "wesn")

    def snap_picture(self):
        self.stop_capturing()
        self.frm_capture.grid_remove()
        global new_pic
        pos_x = (400 - new_pic.width()) / 2
        pos_y = (self.cav_view.winfo_height() - new_pic.height()) / 2

        self.cav_view.coords(self.image_container, pos_x, pos_y)
        self.cav_view.itemconfig(self.image_container,image=new_pic)

        self.frm_view_image.grid(row=1, column=0, sticky= "wesn")

    def update_frame(self):
        
        global new_pic, capture_after_id, query_image
        ret, frame = cap.read()  # Read a frame from the webcam
        if ret:
            # Convert the frame to PIL Image format
            query_image = ImageOps.mirror(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            image6b = self.resize_keep_ratio(query_image, 400, self.cav_capture.winfo_height())

            new_pic = ImageTk.PhotoImage(image6b)
            pos_x = (400 - new_pic.width()) // 2
            pos_y = (self.cav_capture.winfo_height() - new_pic.height()) // 2

            self.cav_capture.coords(self.capture_container, pos_x, pos_y)
            self.cav_capture.itemconfig(self.capture_container,image=new_pic)
            
        # Schedule the next update after a delay
        capture_after_id = self.cav_capture.after(10, self.update_frame)

    def open_camera(self):
        self.lbl_main_left_title.config(text='Captured Image')

        self.frm_main_left.grid_remove()
        self.frm_capture.grid(row=1, column=0, sticky= "wesn")
        # Open the webcam
        global cap
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cav_capture.winfo_height())
        # Start updating the video stream
        self.update_frame()

    def resize_keep_ratio(self, image,  width, height):
        w, h = image.size

        wp = int(float(w * height) / h)
        hp = height

        if wp <= width and hp <=height:
            return image.resize((wp, hp))

        h = int(float(h * width) / w)
        w = width

        return image.resize((w, h))

    def get_embedding(self, model, image):
        x   = keras.utils.img_to_array(image.resize((img_height, img_width)))
        x   = np.expand_dims(x, axis=0)
        x   = preprocess_input(x)
        return model.predict(x, verbose = 0).reshape(-1)

    def recommendation_function(self, image):
        global CNN_model, nn_model
        query_vector = self.get_embedding(CNN_model, image)
        _, indices = nn_model.kneighbors(np.expand_dims(query_vector, axis=0))
        return indices


app = Tkinter_app()
app.mainloop()