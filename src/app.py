import tkinter as tk
import PIL

import numpy as np
from modules.drawbox import Drawbox
from modules.digitAI import DigitAI


def array_pool(array,stride=(2,2),method="mean"):
    """Array pooling
    Convert user drawn image to the size understood by AI
    <arr>: np array, input array to pool (must be multible of stride).
    <stride>: tuple of 2, kernel size in (stride m, stride n).
    <method>: str, 'max' for max-pooling, 
                   'mean' for mean-pooling (default).
    Return <result>: pooled array as integers. 
    """

    M, N = array.shape[:2]
    m,n = stride
    
    newM = M // m
    newN = N // n

    new_shape=(newM,m,newN,n)

    if method=="max":
        result=np.nanmax(array.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(array.reshape(new_shape),axis=(1,3))

    return result.astype("uint8")




def main_window():
    """
    Create the UI 
    """
    global task
    task = None
    
    AI = DigitAI()

    def submit():
        """
        Read the image user has drawn, shrunk it down, and submit it to the AI
        """
        user_img = np.asarray(drawbox.output_image)
        reduced_img = array_pool(user_img,(5,5),"mean")                         # Use Poolin to reduce image size
        reverse_img = np.full((28,28),255,dtype="uint8") - reduced_img      # Reverse the image data 
        AI_image = reverse_img.reshape(-1,28,28,1)                  #Data to be sent for AI   

        re_image = PIL.Image.fromarray(reverse_img, 'L')
        sec_image_canvas.image = PIL.ImageTk.PhotoImage(re_image)     
        sec_image_canvas.itemconfig(sec_image, image = sec_image_canvas.image)              

        AI_reply = AI.recognize(AI_image)                   
        if reply_details.get() == 1:
            reply.set("   {}   \n   [{}]: {}   \n".format(AI_reply[0],AI_reply[1],AI_reply[2]))
        else:
            reply.set("\n   {}   \n".format(AI_reply[2]))
         

    def exit():
        """
        Clear task memory and stop the mainloop
        """
        global task
        if task is not None:
            root.after_cancel(task)
            task = None
        root.destroy()

    """
    Construct UI
    """
    
    width = 570  # canvas width
    height = 300 # canvas height

    root = tk.Tk()
    root.title("Drawing Recognition AI")

    mainframe = tk.Frame(root)
    mainframe.pack(padx=10, pady=10)

    # create a draw box
    draw_width = 140  # canvas width
    draw_height = 140 # canvas height
    drawbox = Drawbox(mainframe, draw_width, draw_height)
    drawbox.grid(row=0, column=0, rowspan=4, columnspan=2)


    # Create box for the AI comment
    reply = tk.StringVar()
    reply.set("\n   Well, hello there! Please draw a number.   \n")
    reply_label = tk.Label(mainframe, height= 3, width=35, textvariable=reply, relief=tk.SUNKEN,bg="white")
    reply_label.grid(row=0, column=3, columnspan = 3,padx=5,pady=5)

    # Show the image that AI "sees" 
    # Label Text
    sec_image_label = tk.Label(mainframe, height= 1, width=20, text="Transformed image")
    sec_image_label.grid(row=3, column=5, padx=5,pady=5)
    # Image
    sec_image_canvas = tk.Canvas(mainframe, width=draw_width/2, height=draw_height/2, relief=tk.GROOVE, bg="white")
    sec_image_canvas.grid(row=4, column=5)
    init_image = PIL.Image.new("L", (int(draw_width/2), int(draw_height/2)), 255)
    sec_image = sec_image_canvas.create_image(draw_width/4,draw_height/4, image=PIL.ImageTk.PhotoImage(init_image))


    # add a button to submit the image
    button_submit = tk.Button(mainframe,text="submit",command=submit)
    button_submit.grid(row=4, column=0,padx=5,pady=5)

    # add a button to clear the image
    button_clear = tk.Button(mainframe,text="clear",command=drawbox.clear)
    button_clear.grid(row=4, column=1,padx=5,pady=5)

    # add a button to exit the program
    button_exit = tk.Button(mainframe,text="EXIT",command=exit)
    button_exit.grid(row=5, column=0,padx=5,pady=5)
    
        # add an AI test button 
    button_test = tk.Button(mainframe,text="Test",command=AI.test)
    button_test.grid(row=5, column=1,padx=5,pady=5)
    
    # add a checkbox to change verbosity of AI reply 
    reply_details = tk.IntVar()
    check_reply = tk.Checkbutton(mainframe,text="Show reply details",variable = reply_details,onvalue = 1, offvalue = 0)
    check_reply.grid(row=5, column=4,padx=5,pady=5)



    root.mainloop()


main_window()






