from extract_cnn_vgg16_keras import VGGNet
import os
import tkinter.filedialog
from tkinter import *
from PIL import Image, ImageTk
from logo import LOGO
import threading
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master, bg='black')
        self.pack(expand=YES, fill=BOTH)
        self.score = StringVar
        self.score = 'score'
        self.window_init()
        self.createWidgets()
        self.model = VGGNet()
        self.logo = LOGO()

    def window_init(self):
        self.master.title('welcome to LOGO Retrieval')
        self.master.bg = 'black'
        self.select_img = ImageTk.PhotoImage(self.img_normalize('input.jpg'))
        self.output_img = ImageTk.PhotoImage(self.img_normalize('matched.jpg'))
        #width, height = self.master.maxsize()
        self.master.geometry("{}x{}".format(1000, 500))

    def createWidgets(self):
        # fm1
        self.fm1 = Frame(self, bg='black')
        self.titleLabel = Label(self.fm1, text="LOGO Retrieval ", font=('微软雅黑', 18), fg="white", bg='black')
        self.titleLabel.pack()
        self.fm1.pack(side=TOP, expand=YES, fill='x', pady=10)

        # fm2
        self.fm2 = Frame(self, bg='black')
        self.fm2_left = Frame(self.fm2, bg='black')
        self.fm2_right = Frame(self.fm2, bg='black')
        self.fm2_left_top = Frame(self.fm2_left, bg='black')
        self.fm2_left_bottom = Frame(self.fm2_left, bg='black')
        self.fm2_right_top = Frame(self.fm2_right, bg='black')
        self.fm2_right_bottom = Frame(self.fm2_right, bg='black')

        self.predictEntry = Entry(self.fm2_left_top, font=('微软雅黑', 9), width='84', fg='#FF4081')
        self.predictButton = Button(self.fm2_left_top, text='select input img', bg='#22C9C9', fg='white',
                                    font=('微软雅黑', 9), width='18', command=self.select_input)
        self.predictButton.pack(side=LEFT)
        self.predictEntry.pack(side=LEFT, fill='both', padx=10)
        self.fm2_left_top.pack(side=TOP, fill='x')

        self.truthEntry = Entry(self.fm2_left_bottom, font=('微软雅黑', 9), width='84', fg='#FF4081')
        self.truthButton = Button(self.fm2_left_bottom, text='get output img', bg='#22C9C9', fg='white',
                                  font=('微软雅黑', 9), width='18', command=self.output)
        self.truthButton.pack(side=LEFT)
        self.truthEntry.pack(side=LEFT, fill='both', padx=10)
        self.fm2_left_bottom.pack(side=TOP, pady=10, fill='x')


        self.fm2_left.pack(side=LEFT, padx=20, pady=20, expand=YES, fill='x')
        self.SetDatabase = Button(self.fm2_right_top, text='CreateDataBase', bg='#FF4081', fg='white',
                                          font=('微软雅黑', 9), width='16', command=self.CreateDataBase)
        self.Img_Score = Label(self.fm2_right_bottom, text=self.score, bg='black', fg='white',
                                      font=('微软雅黑', 9), width='18')
        self.SetDatabase.pack(expand=YES, fill='both')
        self.fm2_right_top.pack(side=TOP, fill='both')
        self.Img_Score.pack(expand=YES, fill='both')
        self.fm2_right_bottom.pack(side=TOP, pady=10, fill='both')
        self.fm2_right.pack(side=RIGHT, padx=60)
        self.fm2.pack(side=TOP, expand=YES, fill="both")

        self.fm3_right = Frame(self, bg='black')
        self.fm3_left = Frame(self, bg='black')

        self.select = Label(self.fm3_left, text='select_img', fg='white', image=self.select_img, compound='top')
        self.select.image = self.select_img
        self.select.pack()

        self.output = Label(self.fm3_right, text='output_img', fg='white', image=self.output_img, compound='top')
        self.output.image = self.output_img
        self.output.pack()

        self.fm3_right.pack(side=RIGHT, expand=YES, fill='x', pady=10)
        self.fm3_left.pack(side=RIGHT, expand=YES, fill='x', pady=10)

    def img_normalize(self, img_path):
        img = Image.open(img_path)
        iw, ih = img.size
        w, h = (320, 320)
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = img.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (320, 320), (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def select_input(self):
        self.file = tkinter.filedialog.askopenfilename()
        self.select_img = ImageTk.PhotoImage(self.img_normalize(self.file))
        self.select.config(image=self.select_img)
        self.select.image = self.select_img

        self.predictEntry.delete(0, END)
        self.predictEntry.insert(0, self.file)


    def output(self):
        self.fm3_right = Frame(self, bg='black')
        print(self.file)
        logo, _, score = self.logo.search_img(query=self.file)
        self.score='score:' + str(score)[0:5]
        self.Img_Score.config(text=self.score)

        self.output_img = ImageTk.PhotoImage(self.img_normalize(logo))
        self.output.config(image=self.output_img)
        self.output.image = self.output_img
        self.truthEntry.delete(0, END)
        self.truthEntry.insert(0, logo)

    def CreateDataBase(self):
        self.logo.create_database(data_path="database/")



if __name__ == '__main__':
    app = Application()
    app.mainloop()