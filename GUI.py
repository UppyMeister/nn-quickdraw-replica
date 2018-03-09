from tkinter import *
from NeuralNetwork import NeuralNetwork

gui = Tk()
gui.geometry("280x280")
gui.title("Doodle Neural Network")

def paint( event ):
    x1, y1 = event.x - 4, event.y - 4
    x2, y2 = event.x + 4, event.y + 4
    canvas.create_oval(x1, y1, x2, y2, fill="#FFFFFF")

canvas = Canvas(gui,width=280,height=280,bg="white")
canvas.pack()
canvas.bind("<B1-Motion>", paint)

mainloop()
