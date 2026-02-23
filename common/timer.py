import time


class Timer():
    def __init__(self, activate):
        self.activate = activate
        self.start_time = []
        self.name = []
        self.buffer = []

    def put(self, name):
        if self.activate:
            self.name.append(name)
            self.start_time.append(time.time())
    
    def get(self):
        if self.activate:
            self.buffer.append(f"{self.name.pop()}: {(time.time() - self.start_time.pop()) * 1000} ms")

    def flush(self):
        if self.activate:
            print('\n'.join(self.buffer))
            self.buffer = []
