import cv2
import numpy as np
import math
from typing import List

NUM_REGIONS = 4
REGION_SIZE = 4
WINDOW_WIDTH = 1
NUM_BINS = 8

class Octave:
    def __init__(self, img: cv2.typing.MatLike):
        self.img = img
        h, w = img.shape
        self.mag = np.zeros((h, w))
        self.angle = np.zeros((h, w))
        self.compute_mag_and_angle()
        
    def compute_mag_and_angle(self):
        h, w = self.img.shape
        img = self.img.astype(np.float32)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                self.mag[i][j] = ((img[i + 1][j] - img[i - 1][j]) ** 2 + (img[i][j + 1] - img[i][j - 1]) ** 2) ** 0.5
                self.angle[i][j] = (360 - math.atan2(img[i, j + 1] - img[i, j - 1], img[i + 1, j] - img[i - 1, j]) * 180 / math.pi) % 360
        
class SIFT:
    def __init__(self):
        self.octaves: List[Octave] = []
        
    def get_kp(self, img):
        corners = cv2.goodFeaturesToTrack(
            img,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=2,
            useHarrisDetector=True,
            k=0.01
        )
        if corners is None:
            return []
        return [cv2.KeyPoint(x=float(x), y=float(y), size=0) for [[x, y]] in corners]

    def gaussian_mask(self, sigma, dx, dy):
        return math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

    def compute_kp_orient(self, kp: List[cv2.KeyPoint]):
        more_kps = []
        for p in kp:
            p.octave = 0
            octave = self.octaves[p.octave]
            img, mag, angle = octave.img, octave.mag, octave.angle
            h, w = img.shape
            scale = 2 ** -p.octave
            
            sigma = p.size * 1.5 * scale
            rad = int(sigma * 3)
            px, py = int(p.pt[0] * scale), int(p.pt[1] * scale)
            
            bins = [0] * 36
            for i in range(max(1, py - rad), min(py + rad + 1, h - 1)):
                for j in range(max(1, px - rad), min(px + rad + 1, w - 1)):
                    m = mag[i][j]
                    a = angle[i][j]
                    bin_id = math.floor(a / 10)
                    bins[bin_id] += m * self.gaussian_mask(sigma, j - px, i - py)
                    
            max_bin_id = np.argmax(bins)
            p.angle = int(max_bin_id * 10 + 5)
            
            bin_max = bins[max_bin_id]
            bins[max_bin_id] = -1
            second_max_bin_id = np.argmax(bins)
            if bins[second_max_bin_id] > bin_max * 0.8:
                more_kps.append(cv2.KeyPoint(p.pt[0], p.pt[1], p.size, int(second_max_bin_id * 10 + 5)))
                
        kp.extend(more_kps)

    def get_val_by_interpolation(self, mat: np.ndarray, x, y): # bilinear interpolation
        h, w = mat.shape
        if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            return 0
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0
        val = (
            mat[y0][x0] * (1 - dx) * (1 - dy)
            + mat[y0][x1] * dx * (1 - dy)
            + mat[y1][x0] * (1 - dx) * dy
            + mat[y1][x1] * dx * dy
        )
        return val
    
    def get_des(self, kp: List[cv2.KeyPoint]):
        des = []
        
        for p in kp:
            desc = np.zeros((NUM_REGIONS, NUM_REGIONS, NUM_BINS))
            
            octave = self.octaves[p.octave]
            scale = 2 ** -p.octave
            img, mag, angle = octave.img, octave.mag, octave.angle
            
            sub_width = 3 * 0.5 * p.size * scale
            radius = np.round(sub_width * (4 + 1) * math.sqrt(2) * 0.5).astype(int)
            
            x, y = int(p.pt[0] * scale), int(p.pt[1] * scale)
            orient = p.angle
            
            h, w = img.shape
            
            sin_orient = math.sin(-orient / 180 * math.pi)
            cos_orient = math.cos(-orient / 180 * math.pi)    
            for yy in range(-radius, radius + 1):
                for xx in range(-radius, radius + 1):
                    y_rot = xx * sin_orient + yy * cos_orient
                    x_rot = xx * cos_orient - yy * sin_orient
                    
                    bin_y = math.floor(y_rot / sub_width) + 2
                    bin_x = math.floor(x_rot / sub_width) + 2
                    
                    if bin_x < 0 or bin_x > 3 or bin_y < 0 or bin_y > 3:
                        continue
                    
                    y_img = y + yy
                    x_img = x + xx
                    
                    if x_img < 1 or x_img >= w - 1 or y_img < 1 or y_img >= h - 1:
                        continue
                    
                    m = self.get_val_by_interpolation(mag, x_img, y_img)
                    a = self.get_val_by_interpolation(angle, x_img, y_img)
                    
                    bin_angle = int((a - p.angle) / 360 * NUM_BINS) % NUM_BINS

                    weight = m * self.gaussian_mask(0.75, y_rot / sub_width, x_rot / sub_width)
                    desc[bin_y, bin_x, bin_angle] += weight

            desc = desc.flatten()
            threshold = 0.2 * np.linalg.norm(desc)
            desc[desc > threshold] = threshold
            desc /= (np.linalg.norm(desc) + 1e-8)

            des.append(desc)
        
        return np.array(des, dtype=np.float32)
    
    # main function
    def detectAndCompute(self, base_img: cv2.typing.MatLike, mask: None = None):
        self.__init__()
        num_octaves = int(round(math.log(min(base_img.shape)) / math.log(2) - 5))
        
        self.octaves.append(Octave(base_img))
        for i in range(num_octaves):
            last_img = self.octaves[i].img
            img = cv2.resize(last_img, (int(last_img.shape[1] / 2), int(last_img.shape[0] / 2))) # add Gaussian blur 1.6?
            self.octaves.append(Octave(img))
        
        all_kps = []
        
        for octave_index, octave in enumerate(self.octaves):
            img = octave.img
            kps: List[cv2.KeyPoint] = self.get_kp(img)
            for kp in kps:
                kp.size = 2 ** (octave_index + 1)
                kp.pt = tuple(np.array(kp.pt) * (2 ** (octave_index)))
                kp.octave = octave_index
            all_kps.extend(kps)
        
        self.compute_kp_orient(all_kps)
        
        des = self.get_des(all_kps)
        
        return all_kps, des