import tkinter as tk
from pandastable import Table, TableModel


class View(tk.Tk):
    def __init__(self, file_callback):
        tk.Tk.__init__(self)

        # window settings
        self.title('Time Series Analysis')
        self.minsize(300, 300)
        self.maxsize(600, 800)
        self.content = tk.Frame()

        # content

        # buttons
        self.content.import_button = tk.Button(text='open file', command=lambda: file_callback(self.get_csv())) \
            .grid(row=0, column=0, sticky='nw')

    def get_csv(self):
        return tk.filedialog.askopenfilename(initialdir="./",
                                             title="Select file",
                                             filetypes=(("csv files", "*.csv"), ("all files", "*.*")))

    def display_dataframe(self, df):
        self.content.df = tk.Label(text=df).grid(row=1, column=0)
