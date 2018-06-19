import tkinter as tk


class View(tk.Tk):
    def __init__(self, file_callback):
        tk.Tk.__init__(self)

        # window settings
        self.title('Time Series Analysis')
        self.minsize(300, 300)
        self.maxsize(600, 800)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side='left')
        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side='right')

        # content

        # buttons
        self.canvas.import_button = tk.Button(text='open file', command=lambda: file_callback(self.get_csv()))
        self.canvas.import_button.pack(side='left')

    def get_csv(self):
        return tk.filedialog.askopenfilename(initialdir="./",
                                             title="Select file",
                                             filetypes=(("csv files", "*.csv"), ("all files", "*.*")))

    def display_dataframe(self, df):
        self.canvas.df = tk.Label(text=df)
        self.canvas.df.pack()
