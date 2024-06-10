import faiss
import numpy as np

import faiss

# class Faiss_:
#     def __init__(self, dim=6):
#         self.dim = dim
#         self.model_faiss = faiss.IndexFlatL2(dim)

#     def faiss_search(self, embs, emb, threshold=0.5, device='cpu', k=1, nprobe=1, train=False):
#         self.model_faiss.add(embs)
#         D, I = self.model_faiss.search(emb, k)
#         similarity_score = 1 / (1 + D[0])  # similarity_score cang lon cang giong nhau
#         self.reset()
#         return I[0], similarity_score  # I[0]: index gia tri giong nhau

#     def reset(self):
#         del self.model_faiss
#         self.model_faiss = faiss.IndexFlatL2(self.dim)


# def euclidean_distance(a, b):
#     """
#     Tính khoảng cách Euclidean giữa hai vector a và b.
#     """
#     return np.sqrt(np.sum((a - b) ** 2))

# def find_nearest_embeddings_euclidean(emb_query, selected_embs, k=3):
#     """
#     Tìm các embeddings gần nhất với emb_query dựa trên khoảng cách Euclidean.

#     Arguments:
#     - emb_query: Embedding vector truy vấn.
#     - selected_embs: Mảng chứa các embedding cần so sánh.
#     - k: Số lượng embedding gần nhất cần tìm.

#     Returns:
#     - nearest_indices: Chỉ số của các embedding gần nhất trong selected_embs.
#     - distances: Khoảng cách tương ứng của các embedding gần nhất.
#     """
#     distances = [euclidean_distance(emb_query, emb) for emb in selected_embs]
#     nearest_indices = np.argsort(distances)[:k]
#     nearest_distances = np.array(distances)[nearest_indices]
    
#     return nearest_indices, nearest_distances

def cosine_distance(A, B):
    # Tính tích vô hướng của A và B
    dot_product = np.dot(A, B)
    
    # Tính norm (độ dài) của A và B
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # Tính khoảng cách cosine
    cosine_dist = 1 - (dot_product / (norm_A * norm_B))
    return cosine_dist

# class Kmeans_faiss:
#     def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4):
#         self.n_clusters = n_clusters
#         self.n_init = n_init
#         self.max_iter = max_iter
#         self.tol = tol

#     def train(self, embs, save_path=None):
#         self.kmeans = faiss.Kmeans(d = embs.shape[1], k = self.n_clusters, niter = self.max_iter, nredo = self.n_init, verbose = True)
#         self.kmeans.train(embs)
#         if save_path:
#             faiss.write_index(self.kmeans.index, save_path)
#         centroids = self.kmeans.centroids
#         distances, indices = self.kmeans.index.search(embs, 1)
#         indices = indices.reshape(-1)
#         return centroids, indices
#     def load(self, path):
#         self.kmeans = faiss.read_index(path)
    
#     def predict(self, emb, embs=None):
#         try:
#             distance, indice = self.kmeans.index.search(emb, 1)
#         except:
#             distance, indice = self.kmeans.search(emb, 1)

#         distances, indices = self.kmeans.search(embs, 1)
#         return indice[0][0], indices.reshape(-1)
    
#     def find_nearest_embeddings(self, emb_query, selected_embs, k=3):
#         indice, indices = self.predict(emb_query, selected_embs)
#         cluster_indices = np.where(indices == indice)[0]
#         cluster_embeddings = selected_embs[cluster_indices]
        
#         nearest_indices, nearest_distances = find_nearest_embeddings_euclidean(emb_query, cluster_embeddings, k)
#         nearest_indices = cluster_indices[nearest_indices]

#         return nearest_indices, nearest_distances


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None

    def fit(self, data):
        # Khởi tạo các centroid ngẫu nhiên từ dữ liệu
        self.centroids = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Gán nhãn cho mỗi điểm dựa trên khoảng cách tới các centroid
            self.labels_ = np.array([self._closest_centroid(point) for point in data])

            # Tính các centroid mới
            new_centroids = self._compute_centroids(data)

            # Kiểm tra hội tụ (nếu các centroid không thay đổi)
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids

    def predict(self, data):
        # Dự đoán nhãn cho dữ liệu mới / cũ
        return np.array([self._closest_centroid(point) for point in data])

    def _closest_centroid(self, point):
        # Tìm centroid gần nhất cho một điểm dựa trên khoảng cách cosine
        distances = [cosine_distance(point, centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def _compute_centroids(self, data):
        # Tính các centroid mới dựa trên các điểm được gán
        centroids = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            points_in_cluster = data[self.labels_ == i]
            if len(points_in_cluster) > 0:
                centroids[i] = np.mean(points_in_cluster, axis=0)
        return centroids

    def find_nearest_embeddings(self, emb_query, selected_embs, k=3):
        indice = self.predict(emb_query)
        indices = self.predict(selected_embs)
        cluster_indices = np.where(indices == indice)[0]
        cluster_embeddings = selected_embs[cluster_indices]
        
        nearest_indices, nearest_distances = self.find_nearest_embeddings_cosine(emb_query, cluster_embeddings, k)
        nearest_indices = cluster_indices[nearest_indices]

        return nearest_indices, nearest_distances
    
    def find_nearest_embeddings_cosine(self, emb_query, cluster_embeddings, k):
        distances = [cosine_distance(emb_query, emb) for emb in cluster_embeddings]
        distances = np.concatenate(distances)
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances =  np.array(distances)[nearest_indices]
        
        return nearest_indices, nearest_distances
    
if __name__ == '__main__':
    # Tạo dữ liệu mẫu
    # fs = Faiss_(dim = 3)

    embs = np.array([[1, 2, 3], [2, 4, 5], [33, 44, 55], [55, 33, 44], [111,212,333], [111, 223, 321]])
    emb = np.array([[1,2,3]])

    # # Gọi hàm faiss_search
    # index, similarity_score = fs.faiss_search(embs, emb, k=3)
    # print(index, similarity_score)

    # nearest_indices, nearest_distances = find_nearest_embeddings_euclidean(emb, embs, 3)
    # print(nearest_indices, nearest_distances)    

    # kf = Kmeans_faiss(n_clusters=3)
    # centroids, indices = kf.train(embs, save_path='weight/kmean.index')
    # kf.load('weight/kmean.index')


    # indice, indices = kf.predict(emb, embs)
    # print(indice, indices)

    # nearest_indices, nearest_distances = kf.find_nearest_embeddings(emb, embs, 3)
    # print(nearest_indices, nearest_distances)

    kmean = KMeans(n_clusters=3)
    kmean.fit(embs)
    np.save('weight/kmean.npy', kmean.centroids)
    centroids = np.load('weight/kmean.npy')
    kmean.centroids = centroids
    nearest_indices, nearest_distances = kmean.find_nearest_embeddings(emb, embs, 3)
    print(nearest_indices, nearest_distances)


    # test data audio
    # import pandas as pd
    # embs = np.empty(shape=[0,6], dtype=np.float32)
    # df = pd.read_csv('datasets_mp3.csv')
    # filenames = df['Trimmed Filename']
    # Descriptions = df['Description']
    # for filename in filenames:
    #     file_npy = filename+'.npy'
    #     emb = np.load(f'data_npy/gallery/{file_npy}')
    #     embs = np.append(embs, [emb], axis=0)


    # from feature_extract import Feature_audio
    # fa = Feature_audio(sr=44100)
    # filename = 'i5Q02YX2VTw_speech_male_0000_0100'
    # audio_query = f"data_wav/gallery/{filename}.wav"
    # emb_query = fa.all_feature(audio_query)
    # emb_query = emb_query.reshape(1, -1)

    # kmeans = KMeans(n_clusters=3)
    

    # kmeans.fit(embs)
    # print(kmeans.centroids)
    # np.save('weight/kmean.npy', kmeans.centroids)
    # centroids = np.load('weight/kmean.npy')
    # kmeans.centroids = centroids
    # nearest_indices, nearest_distances = kmeans.find_nearest_embeddings(emb_query, embs, 3)
    # print(nearest_indices, nearest_distances)

    
