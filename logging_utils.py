import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

orig_stdout = sys.stdout
sys.stdout = Logger("yourlogfilename.txt")
print("Hello world !") # this is should be saved in yourlogfilename.txt