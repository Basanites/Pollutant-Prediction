from datetime import datetime
import os


class Logger:

    def __init__(self, log_location):
        if not os.path.isfile(log_location):
            open(log_location, 'x')
        self.log_location = log_location

    def log(self, string, level=0):
        arrow = ''
        for i in range(1, level + 1):
            arrow += '--'
        if len(arrow):
            arrow += '>'
            output = arrow + ' ' + string
        else:
            output = string

        print(output)
        self.write_to_logfile(output)

    def append_file(self, string, location):
        if not os.path.isfile(location):
            open(location, 'x')

        with open(location, 'a') as file:
            file.write(string)

    def write_to_logfile(self, string):
        with open(self.log_location, 'a') as file:
            file.write(f'{datetime.now()}: {string}\n')
