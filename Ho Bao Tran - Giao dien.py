import tkinter as tk
from tkinter import *
from PIL import ImageTk
from PIL import ImageTk, Image
import sqlite3
from numpy import random
import pyglet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import copy
import cv2
import numpy as np
from keras.models import load_model
import time
# set colours
bg_colour = "#3d6466"

# load custom fonts
pyglet.font.add_file("C:/Users/Admin/Downloads/Ubuntu-Bold.ttf")
pyglet.font.add_file("C:/Users/Admin/Downloads/Shanti-Regular.ttf")
######################################
def clear_widgets(frame):
	# select all frame widgets and delete them
	for widget in frame.winfo_children():
		widget.destroy()
##########################################
# initiallize app with basic settings
root = tk.Tk()
root.title("HỆ THỐNG NHẬN DIỆN NGÔN NGỮ NGƯỜI KHUYẾT TẬT")
root.eval("tk::PlaceWindow . center")
# create a frame widgets
frame1 = tk.Frame(root, width=500, height=600, bg=bg_colour)
frame2 = tk.Frame(root, bg=bg_colour)
###########################################
frame1.tkraise()
	# prevent widgets from modifying the frame
frame1.pack_propagate(False)

logo_img = ImageTk.PhotoImage(file="C:/Users/Admin/Downloads/tran.jpg")
logo_widget = tk.Label(frame1, image=logo_img, bg=bg_colour)
logo_widget.image = logo_img
logo_widget.pack()
lbl = Label(root, text="Hệ thống nhận diện ngôn ngữ người khuyết tật", fg="blue",font=("Arial Bold", 30)).place(x=1, y=80)

for frame in (frame1, frame2):
	frame.grid(row=0, column=0, sticky="nesw")
def nutnhan():
    gesture_names = {0: 'E',
                 1: 'L',
                 2: 'F',
                 3: 'V',
                 4: 'B'}
    model = load_model('C:/Users/Admin/Downloads/handpose.h5')
    def predict_rgb_image_vgg(image):
        image = np.array(image, dtype='float32')
        image /= 255
        pred_array = model.predict(image)
        print(f'pred_array: {pred_array}')
        result = gesture_names[np.argmax(pred_array)]
        print(f'Result: {result}')
        print(max(pred_array[0]))
        score = float("%0.2f" % (max(pred_array[0]) * 100))
        print(result)
        return result, score
    def remove_background(frame):
        fgmask = bgModel.apply(frame, learningRate=learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res
    # Khai bao kich thuoc vung detection region
    cap_region_x_begin = 0.5
    cap_region_y_end = 0.8

    # Cac thong so lay threshold
    threshold = 60
    blurValue = 41
    bgSubThreshold = 50#50
    learningRate = 0

    # Nguong du doan ky tu
    predThreshold= 95

    isBgCaptured = 0  # Bien luu tru da capture background chua

    # Camera
    camera = cv2.VideoCapture(0)
    camera.set(10,200)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)
    while camera.isOpened():
        # Doc anh tu webcam
        ret, frame = camera.read()
        # Lam min anh
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        # Lat ngang anh
        frame = cv2.flip(frame, 1)

        # Ve khung hinh chu nhat vung detection region
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                    (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    # Neu ca capture dc nen
        if isBgCaptured == 1:
            # Tach nen
            img = remove_background(frame)

            # Lay vung detection
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            # Chuyen ve den trang
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

            cv2.imshow('original1', cv2.resize(blur, dsize=None, fx=0.5, fy=0.5))

            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cv2.imshow('thresh', cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5))

            if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0])>0.2):
                # Neu nhu ve duoc hinh ban tay
                if (thresh is not None):
                    # Dua vao mang de predict
                    target = np.stack((thresh,) * 3, axis=-1)
                    target = cv2.resize(target, (224, 224))
                    target = target.reshape(1, 224, 224, 3)
                    prediction, score = predict_rgb_image_vgg(target)
                    # Neu probality > nguong du doan thi hien thi
                    print(score,prediction)
                    if (score>=predThreshold):
                        cv2.putText(frame, "Sign:" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        thresh = None

        # Xu ly phim bam
        k = cv2.waitKey(10)
        if k == ord('q'):  # Bam q de thoat
            break
        elif k == ord('b'):
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 10, lineType=cv2.LINE_AA)
            time.sleep(2)
            print('Background captured')
        elif k == ord('r'):

            bgModel = None
            isBgCaptured = 0
            cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255),10,lineType=cv2.LINE_AA)
            print('Background reset')
            time.sleep(1)
        cv2.imshow('original', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))
    cv2.destroyAllWindows()
    camera.release()


                

            
    






btn = Button(root, text="Bấm để nhận diện", bg="white",font=("Arial Bold", 20), fg="blue",height= 1, width=52,command=nutnhan)
#Thiết lập vị trí của nút nhấn có màu nền và màu chữ
btn.grid(row=1, column=0)

# run app
root.mainloop()