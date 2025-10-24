import cv2
import numpy as np
from matcher import LSHMatcher, NNMatcher
from matplotlib import pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']

NUM_LOOP = 100

def generate_feature_vector(img: cv2.typing.MatLike):
    h, w, _ = img.shape
    vector = []
    for i in range(4):
        crop_img = img[(i // 2) * h // 2 : (((i // 2) + 1) * h) // 2, (i % 2) * w // 2 : (((i % 2) + 1) * w) // 2]
        totals = crop_img.sum(axis=(0, 1)).astype('float64')
        vector.extend(totals / totals.sum())
    return np.array(vector)

def display_best_match(match: cv2.typing.MatLike, target: cv2.typing.MatLike, title: str = ''):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title, fontsize=16)
    axes[0].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    axes[0].set_title("目标图片", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(match, cv2.COLOR_BGR2RGB))
    axes[1].set_title("最佳匹配", fontsize=12)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f'output/result-{title.split()[0]}.png', dpi=600)

def main():
    dataset_paths = [f"images/dataset/{i + 1}.{'jfif' if i + 1 == 48 else 'jpg'}" for i in range(50)]
    dataset = [cv2.imread(path) for path in dataset_paths]
    target = cv2.imread("images/target.jpg")
    
    dataset_vectors = [generate_feature_vector(d) for d in dataset]
    target_vector = generate_feature_vector(target)
    
    run_once(dataset_vectors, target_vector, dataset, target)
    test(dataset_vectors, target_vector, dataset, target)
    
def run_once(dataset_vectors, target_vector, dataset, target):
    projectors = [
        [1, 4, 8, 10, 14, 19],
    ]
    
    print('Performing LSH Match:')
    matcher_LSH = LSHMatcher(dataset_vectors, target_vector)
    matcher_LSH.set_projectors([np.array(p) for p in projectors])
    time_LSH, best_match_LSH = matcher_LSH.match()
    display_best_match(dataset[best_match_LSH], target, title="LSH Best Match")
    print(f"Time used: {time_LSH * 1000:.4f} ms")
    
    print('Performing NN Match:')
    matcher_NN = NNMatcher(dataset_vectors, target_vector)
    time_NN, best_match_NN = matcher_NN.match()
    display_best_match(dataset[best_match_NN], target, title="NN Best Match")
    print(f"Time used: {time_NN * 1000:.4f} ms")
    
def test(dataset_vectors, target_vector, dataset, target):
    matchers = {
        "LSH": LSHMatcher(dataset_vectors, target_vector),
        "NN": NNMatcher(dataset_vectors, target_vector)
    }
    
    # test projector size
    avg_times_LSH = []
    for proj_size in range(24):
        times = []
        for _ in range(NUM_LOOP):
            projectors = [random.choices(range(24), k=proj_size)]
            matcher_LSH: LSHMatcher = matchers["LSH"]
            matcher_LSH.set_projectors(projectors)
            time_, match_ = matcher_LSH.match()
            times.append(time_ * 1000) # ms
        avg_time = np.sum(times) / NUM_LOOP
        avg_times_LSH.append(avg_time)
    
    times_NN = []
    for _ in range(NUM_LOOP):
        matcher_NN: NNMatcher = matchers["NN"]
        time_, match_ = matcher_NN.match()
        times_NN.append(time_ * 1000) # ms
    avg_times_NN = [np.sum(times_NN) / NUM_LOOP] * 24
    
    plt.figure()
    plt.plot(range(24), avg_times_LSH, label='LSH')
    plt.plot(range(24), avg_times_NN, label='NN')
    plt.legend()
    plt.xlabel('投影集合元素数量')
    plt.ylabel('单次运行平均用时（毫秒）')
    plt.savefig(f'../doc/images/time-n.png', dpi=600)
    
    # test dataset size
    avg_times = {}
    for name in matchers:
        avg_times[name] = []
    for __ in range(10):
        matchers = {
            "LSH": LSHMatcher(dataset_vectors[:5 * __ + 5], target_vector),
            "NN": NNMatcher(dataset_vectors[:5 * __ + 5], target_vector)
        }
        times = {}
        for name in matchers:
            matcher = matchers[name]
            times[name] = []
            for _ in range(NUM_LOOP):
                if name == "LSH":
                    projectors = [random.choices(range(24), k=7)]
                    matcher.set_projectors(projectors)
                time_, match_ = matcher.match()
                times[name].append(time_ * 1000) # ms
        for name in times:
            avg_times[name].append(np.sum(times[name]) / NUM_LOOP)
    
    plt.figure()
    plt.plot(range(5, 51, 5), avg_times["LSH"], label='LSH')
    plt.plot(range(5, 51, 5), avg_times["NN"], label='NN')
    plt.legend()
    plt.xlabel('数据集大小')
    plt.ylabel('单次运行平均用时（毫秒）')
    plt.savefig(f'../doc/images/time-size.png', dpi=600)
    
if __name__ == "__main__":
    main()