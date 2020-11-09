from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
#from numpy import asarray

class Drawbox(tk.Canvas):
    def __init__(self,master,width,height):
        super().__init__(master, width=width, height=height, cursor="dot", bg="white")

        white = 255 # canvas back

        # create an empty PIL image and draw object to draw on
        self.output_image = PIL.Image.new("L", (width, height), 255)
        self.draw = ImageDraw.Draw(self.output_image)

        # Follow mouse commands
        self.bind("<B1-Motion>", self._paint)

    def _paint(self,event):
        """
        Paint when mouse is active
        """
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.create_oval(x1, y1, x2, y2, fill="black",width=7)                 # Draws user visible shapes in ovals
        self.draw.line([x1-3, y1-3, x2+3, y2+3],fill="black",width=9)       # Modifies the data of the imge

    def clear(self):
        """
        Fill canvs with white
        """
        self.create_rectangle((0, 0, 140, 140), fill="white")                   # Clear visible image
        self.draw.rectangle((0, 0, 140, 140), fill="white")                      # Clear object data
