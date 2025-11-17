import cv2
import img_io as iio
from sift import SIFT

def match(sift, dataset, target):
    kp_target, des_target = sift.detectAndCompute(target, None)
    bf = cv2.BFMatcher()
    result = []
    for img in dataset:
        kp, des = sift.detectAndCompute(img, None)
        matches = bf.knnMatch(des_target, des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        matched_img = cv2.drawMatches(target, kp_target, img, kp, good_matches, img, flags=2)
        result.append(matched_img)
    return result

def main(useOpenCV: bool):
    dataset, target = iio.load_dataset_and_target()
    sift = cv2.SIFT_create() if useOpenCV else SIFT()
    result = match(sift, dataset, target)
    iio.save_output(result, suffix='opencv' if useOpenCV else None)

if __name__ == "__main__":
    main(True) # OpenCV
    main(False) # mine