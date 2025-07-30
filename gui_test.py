import tkinter as tk
import cv2#open cv import kiya
import PIL.Image , PIL.ImageTk#Pillow libraries
import time#frame update me delay k liye

class VideoStreamApp:
    def __init__(self, window , window_title , video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():#to check camera open che k nai
            raise ValueError("Camera isn't opened! , Check the camera status")


        self.is_camera_on = False  # Start with camera initially off by default


        #Canvas banavyu tya video dekhase
        #Size camera feed nu resolution na according set kryu
        self.canvas = tk.Canvas(window , width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()


        #Button to start/stop camera add kriye
        self.btn_start_stop = tk.Button(window , text = "Start/Stop Camera" , width=50 ,command= self.toggle_camera)
        self.btn_start_stop.pack(anchor=tk.CENTER , expand = True)

        self.delay = 15#miliseconds ma delay  , aa value frame rate control kre , lower is faster


        self.update_video_feed()#Video feed ne update krvanu chalu kryu

        self.window.mainloop()#Tkinter event loop ne start kryu



    def toggle_camera(self):
        #aa function ne pchi update krsu amna khali aa ek placeholder che
        print("Camera Toggle Button Clicked")
        #ahiyaa apde camera on/off nu logic lakhi skiye
        if self.is_camera_on:
            if self.vid.isOpened():
                self.vid.release()

            self.is_camera_on = False
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas.winfo_width()/2 , self.canvas.winfo_height()/2,text="Camera Off" , fill="red" , font=("Arial",24))
            self.btn_start_stop.config(text="Start Camera")#button nu text badli didhu
            print("Camera is OFF")

        else:#when camera is  off
            self.vid = cv2.VideoCapture(self.video_source)
            if not self.vid.isOpened():
                self.is_camera_on = False
                self.btn_start_stop.config(text="Start Camera")
                self.canvas.delete("all")
                self.canvas.create_text(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2, text="Camera Error!", fill="red", font=("Arial", 24))
                raise ValueError("Camera isn't open!please check the camera status")
            else:
                self.is_camera_on = True
                self.btn_start_stop.config(text="Stop Camera")
                print("Camera is ON")



    def update_video_feed(self):
        #to read a frame from Camera
        ret , frame = self.vid.read()#ret True/False batave ane frame image aape

        if ret:
            #OpenCV(BGR) image ne PIL(RGB) image ma convert kriye
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0,0, image = self.photo , anchor = tk.NW)


        self.window.after(self.delay , self.update_video_feed)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoStreamApp(root , "Live Camera Feed")
# root = tk.Tk()
# root.title("Mera Pehla GUI")
# root.geometry("400x300")
#
# label = tk.Label(root , text = "Hello , Gemini User")
# label.pack()
#
# def button_clicked():
#     print("Button clicked!")
#     label.config(text="Button Clicked! Text Changed!")
#
# button = tk.Button(root ,  text="Click Me!" , command=button_clicked)
# button.pack()
# root.mainloop()

