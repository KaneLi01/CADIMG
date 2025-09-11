import pickle

def get_pkl(pkl_path, key):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data[key]
