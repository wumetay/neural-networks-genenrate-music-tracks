from tkinter import *
from tkinter import filedialog
from traceback import *
from win32com.client import Dispatch
import time,eyed3,threading
import mido
from PIL import Image, ImageTk, ImageSequence
from mido import MidiFile

from predict import generate


name = []

def openfile(index = [1]):
  global total,name

  filenames = filedialog.askopenfilenames(title = "Composer",filetypes =[("MIDI文件","*.mid"),("mp3文件","*.mp3"),("WMA文件","*.wma"),("WAV文件","*.wav")])
  print(filenames)
  
  if filenames:
    for i in range(len(filenames)):
      
      media = wmp.newMedia(filenames[i])
      wmp.currentPlaylist.appendItem(media)
      print(filenames[i][-3:])
      if filenames[i][-3:] == 'mid':
        mid = MidiFile(filenames[i])
        total = int(mid.length)
        minute = int(mid.length)//60
        sec = int(mid.length)%60
        length = int(mid.length)
      else:
          coco = eyed3.load(filenames[i])#eyed3模块读取mp3信息
          total = int(coco.info.time_secs)
          minute = int(coco.info.time_secs)//60
          sec = int(coco.info.time_secs)%60
          length = int(coco.info.time_secs)
      

      name = filenames[i].split("/")

      i =index[-1]
      list_name.insert(END,str(i)+"."+name[-1])
      list_name.insert(END," "*6)
      if sec >=10:
        list_name.insert(END,"0%d:%d" %(minute,sec)+ "\n")
      else:
        list_name.insert(END,"0%s:0%d" %(minute,sec)+ "\n")
      i = i +1
      index.append(i)
def play(event = None):
  #root.title("%s" % name[-1]),使用wmp.currentMedia.name更好,在per函数中
  per_thread = threading.Thread(target = per)
  per_thread.daemnon = True
  wmp.controls.play()
  per_thread.start()
  #print(wmp.currentMedia.duration)#放到暂停那里居然可以用,而这里不行


def per():
  global total
  while wmp.playState !=1:
    progress_scal.set(int(wmp.controls.currentPosition))
    progress_scal.config(label = wmp.controls.currentPositionString)
    progress_scal.config(to = total,tickinterval = 50)
    time.sleep(1)
    root.title("%s" % wmp.currentMedia.name)

def stop():
  wmp.controls.stop()
def pause(event = None):
  wmp.controls.pause()

def uselist():
    pass
def fullscr():
    pass
def exitit():
  root.destroy()
def Previous_it():
  wmp.controls.previous()
def Next_it():
  wmp.controls.next()
def Volume_ctr(none):
  wmp.settings.Volume = vio_scale.get()
def Volume_add(i=[0]):
  wmp.settings.Volume =wmp.settings.Volume+5
  i.append(wmp.settings.Volume)
  vio_scale.set(wmp.settings.Volume)
def Volume_minus(i=[0]):
  wmp.settings.Volume = wmp.settings.Volume -5
  i.append(wmp.settings.Volume)
  vio_scale.set(wmp.settings.Volume)
def Scale_ctr(none):

  wmp.controls.currentPosition = var_scale.get()
  print(wmp.currentMedia.duration)
def Clear_list():
  wmp.currentPlaylist.clear()
  list_name.delete(1.0,END)
  name = []
  index = []
def List_random():
  wmp.settings.setMode("shuffle",True)
  play()
def List_loop():
  wmp.settings.setMode("loop",True)
  play()
    

    
    
root =Tk()
root.title('Composer')
wmp = Dispatch("WMPlayer.OCX")

canvas = Canvas(root,width =240,height = 240,bg = "white")
img = PhotoImage(file = 'D:/program/python/Lib/mylib/Classical-Piano-Composer-master1/Classical-Piano-Composer-master/Composer.gif')
canvas.create_image((120,120),image = img)
canvas.place(x=200,y=0)
canvas.coords(img,0,0)
canvas.grid(row =0,column = 0,sticky = "nw",rowspan =2)

#分解gif并逐帧显示
def pick(event):
    global a,flag   
    while 1:
        im = Image.open('composer.gif')
        # GIF图片流的迭代器
        iter = ImageSequence.Iterator(im)
        #frame就是gif的每一帧，转换一下格式就能显示了
        for frame in iter:
            pic=ImageTk.PhotoImage(frame)
            canvas.create_image((120,120), image=pic)
            time.sleep(0.1)
            root.update_idletasks()  #刷新
            root.update()

            
# 第4步，在图形界面上创建一个标签label用以显示并放置
var1 = StringVar()  # 创建变量，用var1用来接收鼠标点击具体选项的内容
l = Label(root, bg='white', fg='black',font=('Arial', 12), width=10, textvariable=var1)
l.grid(row =2,column =2)
 
# 第6步，创建一个方法用于按钮的点击事件
def print_selection():
    value = lb.get(lb.curselection())   # 获取当前选中的文本
    var1.set(value)  # 为label设置值
    print(var1.get())
 
# 第5步，创建一个按钮并放置，点击按钮调用print_selection函数
b1 = Button(root, text='selection instrument', width=18, height=2, command=print_selection)
b1.grid(row =3,column =2)
 
# 第7步，创建Listbox并为其添加内容
var2 = StringVar()
var2.set(('Piano')) # 为变量var2设置值
# 创建Listbox
lb = Listbox(root, listvariable=var2)  #将var2的值赋给Listbox
# 创建一个list并将值循环添加到Listbox控件中
lb.grid(row =4,column =2)

var3 = StringVar()
l2 = Label(root, bg='white', fg='black', width=20)
l2.grid(row =8,column =2)

def selection(v):
    var3.set(v)
    l2.config(text='duration: ' + v)
s = Scale(root, label='select music duration', from_=0, to=10, orient=HORIZONTAL, length=200, showvalue=0,tickinterval=2, resolution=0.01, command=selection)
s.grid(row =9,column =2)
#生成音乐
def generate_music():  
  print("Generate already")
  generate( )
    
progress_lab = LabelFrame(root,text = "Play schedule")
progress_lab.grid(row =2,column =0,sticky = "we",rowspan = 2)
var_scale = DoubleVar()
progress_scal = Scale(progress_lab,orient = HORIZONTAL,showvalue = 0,length =180,variable = var_scale)
progress_scal.bind("<Button-1>",pause)
progress_scal.bind("")
progress_scal.bind("<ButtonRelease-1>",play)
progress_scal.grid(row =3,column =0)

modee_lab = LabelFrame(root,text = "Play mode")
modee_lab.grid(row =4,column =0,rowspan =2,sticky = "ws")
var_mode = IntVar()
randomradio = Radiobutton(modee_lab,variable = var_mode,value = 1,text ="Random mode",command =List_random )
randomradio.grid(row =4,column =2)
inturnradio = Radiobutton(modee_lab,variable = var_mode,value =2,text= "Order mode",command = play)
inturnradio.grid(row=4,column =3)
alloop = Radiobutton(modee_lab,variable = var_mode,value =2,text = "All circulation mode",command = List_loop)
alloop.grid(row =5,column = 2)
sinloop = Radiobutton(modee_lab,variable = var_mode,value =3,text = "Single circulation mode")
sinloop.grid(row =5,column =3)
previous_play = Button(modee_lab,text = "Last",command = Previous_it)
previous_play.grid(row =6,column =2,rowspan =2,pady =10)
next_play = Button(modee_lab,text = "Next",command = Next_it)
next_play.grid(row =6,column =3,rowspan =2,pady =10)

var_volume = IntVar()
vioce_lab = LabelFrame(root,text = "Volume control")
vioce_lab.grid(row =8,column =0,sticky = "wes")
vio_scale = Scale(vioce_lab,orient = HORIZONTAL,length =170,variable = var_volume,command =Volume_ctr)
vio_scale.set(30)
vio_scale.grid(row =8,column =0)
vio_plus = Button(vioce_lab,width =8,text = "Volume+",command =Volume_add)
vio_plus.grid(row =9,column =0,sticky = "w")
vio_minus = Button(vioce_lab,width =8,text ="Volume-",command = Volume_minus)
vio_minus.grid(row =9,column =0,sticky ="e")

ctr_lab = LabelFrame(root,text = "Play control",height =130)
ctr_lab.grid(row =0,column =1,rowspan =12,sticky = "ns")
btn_open = Button(ctr_lab,text ="Open file",width =12,command = openfile)
btn_open.grid(row=0,column =1)
btn_play = Button(ctr_lab,text ="Generate midi",width =12,command = generate_music)
btn_play.grid(row =1,column =1,pady =5)
btn_stop = Button(ctr_lab,text ="Play",width =12,command = play)
btn_stop.grid(row =2,column =1,pady =5)
btn_pause = Button(ctr_lab,text ="Pause",width =12,command = pause)
btn_pause.grid(row =3,column =1,pady =5)

btn_playlist = Button(ctr_lab,width =12,text ="New playlist",command = uselist)
btn_playlist.grid(row =4,column =1,pady =5)


listimport = Button(ctr_lab,width =12,text = "Import list")
listimport.grid(row =6,column =1,sticky ="nw",pady =5)
listexport = Button(ctr_lab,width =12,text = "Export list")
listexport.grid(row =7,column =1,sticky = "nw",pady =5)
listdel_all = Button(ctr_lab,width =12,text = "Clear list",command = Clear_list)
listdel_all.grid(row =8,column =1,sticky ="nw",pady =5)
listdel_sel= Button(ctr_lab,width =12,text = "Delete", command = exitit )
listdel_sel.grid(row =12,column =1,sticky = "nw",pady =5)
savelist_btn = Button(ctr_lab,width =12,text = "Save list")
savelist_btn.grid(row=9,column =1)
min_btn = Button(ctr_lab,width =12,text = "Minimize",command = root.iconify)
min_btn.grid(row =13,column =1)

##time_lab= Label(root,width =20,height =2,text ="现在时间为:")
##time_lab.grid(row =12,column =0,sticky = "nw",pady =5)
time_text= Text(root,width =30,height =3,foreground ="green")
time_text.grid(row =10,column =0,sticky = "nw",pady =5)

list_name = Text(root,height =18,width =50)
list_name.grid(row =0,column =2,sticky = "n",rowspan =6)
canvas.bind("<Button-1>",pick)  #这个事件是鼠标进入组件，用什么事件不重要，这里只是演示

root.mainloop()