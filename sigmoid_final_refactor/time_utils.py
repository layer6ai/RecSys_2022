import time


class Timer(object):
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        print(f'{self.name} time elapsed: {time.time() - self.start:.2f} seconds...')
