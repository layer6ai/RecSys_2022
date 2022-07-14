import pickle


def to_pickle(obj, path: str):
    with open(path, 'wb+') as file:
        pickle.dump(obj, file=file)


def read_pickle(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file=file)
