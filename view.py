import tkinter as tk

class View(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # window settings
        self.title('Time Series Analysis')
        self.minsize(300, 300)

        # content