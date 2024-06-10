from search import KMeans
import numpy as np
import pandas as pd
from feature_extract import Feature_audio

# fs = Faiss_(dim = 6)
# kf = Kmeans_faiss(n_clusters=3)
fa = Feature_audio(sr=44100)

embs = np.empty(shape=[0,6], dtype=np.float32)

df = pd.read_csv('datasets_mp3.csv')
filenames = df['Trimmed Filename']
Descriptions = df['Description']
for filename in filenames:
    file_npy = filename+'.npy'
    emb = np.load(f'data_npy/gallery/{file_npy}')
    embs = np.append(embs, [emb], axis=0)

def query_audio(filename):
    audio_query = f"data_wav/gallery/{filename}.wav"
    print(f'audio_query: {audio_query}')
    print('-------------------')
    emb_query = fa.all_feature(audio_query)
    emb_query = emb_query.reshape(1, -1)

    # print("Faiss search")
    # indexs, similarity_score = fs.faiss_search(embs, emb_query, k= 3)

    # for i, index in enumerate(indexs) :
    #     print('Descriptions: ', Descriptions[index])
    #     print(filenames[index])
    #     print(similarity_score[i])
    #     print('-------------------')

    # print('-'*20)
    # print('-'*20)
    # print('-'*20)
    # print("Euclidean search")
    # nearest_indices, nearest_distances = find_nearest_embeddings_euclidean(emb_query, embs, 3)
    # for i, index in enumerate(nearest_indices) :
    #     print('Descriptions: ', Descriptions[index])
    #     print(filenames[index])
    #     print(nearest_distances[i])
    #     print('-------------------')

    # print('-'*20)
    # print('-'*20)
    # print('-'*20)
    # print("Kmeans-faiss search")
    # kf.train(embs, save_path='weight/kmean.index')
    # kf.load('weight/kmean.index')
    # nearest_indices, nearest_distances = kf.find_nearest_embeddings(emb_query, embs, 3)
    # for i, index in enumerate(nearest_indices) :
    #     print('Descriptions: ', Descriptions[index])
    #     print(filenames[index])
    #     print(nearest_distances[i])
    #     print('-------------------')

    print('-'*20)
    print('-'*20)
    print('-'*20)
    print("Kmeans search")
    kmean = KMeans(n_clusters=3)

    kmean.fit(embs) # train

    np.save('weight/kmean.npy', kmean.centroids) # save centroids
    centroids = np.load('weight/kmean.npy') # load centroids
    kmean.centroids = centroids # set centroids

    nearest_indices, nearest_distances = kmean.find_nearest_embeddings(emb_query, embs, 3)
    
    for i, index in enumerate(nearest_indices) :
        print('Descriptions: ', Descriptions[index])
        print(filenames[index])
        print(nearest_distances[i])
        print('-------------------')


if __name__ == '__main__':
    query_audio('i5Q02YX2VTw_speech_male_0000_0100') # 00:00 01:00


