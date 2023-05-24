#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# In[2]:


mainwindow = Tk()
img = 0
page_title = "Image Clasification"
img_path = 0
img_categories = "airplane\nbicycle\nbus\nmotorbike\ntruck"
category = "The Result is : "


# In[4]:


def picture_choose_btn_comm():
    global img, img_path, category
    img_path = askopenfilename(filetypes=[("All Files", ".*")])
    if not img_path:
        return
    img_open = Image.open(img_path)
    if img_open.width > 500 and img_open.height > 500:
        img_open.thumbnail((500, 500))
    img = ImageTk.PhotoImage(img_open)

    canvas = Canvas(master=img_frm, width=500, height=500)
    canvas.grid(row=0, column=1)
    canvas.create_image(250, 250, anchor=CENTER, image=img)

    image = tf.keras.preprocessing.image.load_img(path=img_path, target_size=(200, 200))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    Model = tf.keras.models.load_model('vehicle_classification (1).h5')
    y_pred = Model.predict(input_arr)
    classes = ['airplane', 'bicycle','bus','motorbike','truck']
    category = classes[np.argmax(y_pred)]
    result_lbl['text'] = category


# In[5]:


col1_frm = Frame(master=mainwindow)
col1_frm.grid(row=0, column=1)

col0_frm = Frame(master=mainwindow)
col0_frm.grid(row=0, column=0)

result_lbl = Label(master=col1_frm, text=category, relief='raised', pady=3, padx=5, borderwidth=3)
result_lbl.grid(row=0, column=0, pady=10)

img_frm = Frame(master=col1_frm, relief='groove', borderwidth=2, width=500, height=500)
img_frm.grid(row=1, column=0)

categories_lbl_frm = Frame(master=col0_frm, relief='ridge', borderwidth=1, width=100, height=100)
categories_lbl = Label(master=categories_lbl_frm, text=f"Choose from the Following Categories : \n{img_categories}",
                       pady=3, padx=5)
categories_lbl_frm.grid(row=0, column=0)
categories_lbl.grid()
picture_choose_btn = Button(master=col0_frm, text='Upload Picture Here.', width=20, command=picture_choose_btn_comm)
picture_choose_btn.grid(row=1, column=0, pady=10)


# In[6]:


mainwindow.columnconfigure(0, minsize=100, weight=1)
mainwindow.rowconfigure(0, minsize=100, weight=1)
mainwindow.columnconfigure(1, minsize=500, weight=1)
mainwindow.rowconfigure(0, minsize=500, weight=1)
mainwindow.geometry("800x600")
mainwindow.title(page_title)

mainwindow.mainloop()


# In[ ]:




