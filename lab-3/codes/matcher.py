import numpy as np
from typing import List
import time
import matplotlib.pyplot as plt

def get_distance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a - b)

class Matcher:
    def __init__(self, dataset_vectors: List[np.ndarray], target_vector: List[np.ndarray]):
        self.dataset_vectors = dataset_vectors
        self.target_vector = target_vector

class LSHMatcher(Matcher):
    def __init__(self, dataset_vectors: List[np.ndarray], target_vector: List[np.ndarray]):
        Matcher.__init__(self, dataset_vectors, target_vector)
        self.projector: np.ndarray[np.int64] = None
        
    def set_projectors(self, projectors: List[np.ndarray[np.int64]]):
        self.projectors = np.array(projectors).astype(np.int64)
    
    def match(self):
        if self.projectors is None:
            return 0, []
        
        candidates = set()
        
        # start_time = time.perf_counter()
        
        dataset_hamming = [self._get_hamming(vec) for vec in self.dataset_vectors]
        target_hamming = self._get_hamming(self.target_vector)
        
        # print(f"Computing hamming used: {(time.perf_counter() - start_time) * 1000:.4f} ms")
        # start_time = time.perf_counter()
        
        start_time = time.perf_counter()
        
        for proj in self.projectors:
            dataset_hashes = [self._get_hash_vector(ham, proj) for ham in dataset_hamming]
            target_hash = self._get_hash_vector(target_hamming, proj)
            
            for index, hash in enumerate(dataset_hashes):
                if hash != target_hash:
                    continue
                candidates.add(index)
        
        if not candidates:
            return 0, []
        
        # print(f"Finding candidates used: {(time.perf_counter() - start_time) * 1000:.4f} ms")
        # start_time = time.perf_counter()
        
        distances = []
        for c in candidates:
            dst = get_distance(self.target_vector, self.dataset_vectors[c])
            distances.append((c, dst))
        distances.sort(key=lambda x:x[1])
        
        # print(f"Computing best match used: {(time.perf_counter() - start_time) * 1000:.4f} ms")
        
        return time.perf_counter() - start_time, distances[0][0]
    
    def _quantize_feature_vector(self, feature_vector: np.ndarray):
        v = feature_vector.copy()
        v[v >= 0.6] = 2
        v[v < 0.3] = 0
        v[(0.3 <= v) & (v < 0.6)] = 1
        return v.astype(np.int64)
    
    def _get_hamming(self, feature_vector: np.ndarray):
        hamming = []
        for p in self._quantize_feature_vector(feature_vector):
            hamming.extend([1] * p + [0] * (2 - p))
        return np.array(hamming)
    
    def _get_hash_vector(self, hamming: np.ndarray, subset: tuple):
        return tuple(hamming[subset])
    
class NNMatcher(Matcher):        
    def match(self):
        start_time = time.perf_counter()
        distances = []
        for index, d_vec in enumerate(self.dataset_vectors):
            dst = get_distance(self.target_vector, d_vec)
            distances.append((index, dst))
        distances.sort(key=lambda x:x[1])
        return time.perf_counter() - start_time, distances[0][0]