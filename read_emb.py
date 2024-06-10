import numpy as np
import os
emb = np.load('data_npy\gallery\i5Q02YX2VTw_speech_male_0000_0100.npy')
print(emb)

centroids = np.load('weight\kmean.npy')
print(centroids)

path_embs = os.listdir('data_npy\gallery')
with open('embs.txt', 'w') as f:
    for path_emb in path_embs:
        emb = np.load(f'data_npy\gallery\{path_emb}')
        f.write(str(emb) + '\n')