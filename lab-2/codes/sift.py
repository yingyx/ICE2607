import cv2
import numpy as np
import math
from typing import List
import logging
import sys
from functools import cmp_to_key
import debug

NUM_REGIONS = 4
REGION_SIZE = 4
NUM_BINS = 8

class Octave:
    def __init__(self, img: cv2.typing.MatLike):
        self.img = img
        self.mag = np.zeros(self.img.shape)
        self.angle = np.zeros(self.img.shape)
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
            minDistance=10,
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
            octave = self.octaves[p.octave]
            img, mag, angle = octave.img, octave.mag, octave.angle
            h, w = img.shape
            scale = 2 ** -p.octave
            
            sigma = p.size * 1.5 * scale
            rad = int(sigma * 3)
            px, py = int(p.pt[0] * scale), int(p.pt[1] * scale)
            
            # fill bins by orientation
            bins = [0] * 36
            for i in range(max(1, py - rad), min(py + rad + 1, h - 1)):
                for j in range(max(1, px - rad), min(px + rad + 1, w - 1)):
                    m = mag[i][j]
                    a = angle[i][j]
                    bin_id = math.floor(a / 10)
                    bins[bin_id] += m * self.gaussian_mask(sigma, j - px, i - py)
            
            # find peaks and interpolate accurate orientation
            bin_max = max(bins)
            peaks = np.where(np.logical_and(bins > np.roll(bins, 1), bins > np.roll(bins, -1)))[0]
            for peak_index in peaks:
                val = bins[peak_index]
                if val >= 0.8 * bin_max and val < bin_max:
                    val_left = bins[(peak_index - 1) % 36]
                    val_right = bins[(peak_index + 1) % 36]
                    intp_peak_index = (peak_index + 0.5 * (val_left - val_right) / (val_left - 2 * val + val_right)) % 36
                    more_kps.append(cv2.KeyPoint(*p.pt, p.size, int(intp_peak_index * 10 + 5)))
            
            max_bin_id = np.argmax(bins)
            p.angle = int(max_bin_id * 10 + 5)
            
            logging.debug(f"Keypoint ({p.pt[0]}, {p.pt[1]}) angle {p.angle} determined with radius {rad} and sigma {sigma}.")
                
        kp.extend(more_kps)
        
        logging.info(f'Assigned different orientations to {len(more_kps)} ({round(len(more_kps) / (len(kp) - len(more_kps)) * 100, 1)}%) keypoints.')

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
    
    def compare_kp(self, a: cv2.KeyPoint, b: cv2.KeyPoint):
        if a.pt[0] != b.pt[0]:
            return a.pt[0] - b.pt[0]
        if a.pt[1] != b.pt[1]:
            return a.pt[1] - b.pt[1]
        return b.size - a.size
    
    def remove_duplicate_kp(self, kp: List[cv2.KeyPoint]):
        if not len(kp):
            return []
        unique_kps = [kp[0]]
        kp.sort(key=cmp_to_key(self.compare_kp))
        i = 0
        for p in kp[1:]:
            if kp[i].pt == p.pt and kp[i].size > p.size:
                    continue
            i += 1
            unique_kps.append(p)
            
        logging.info(f'Removed {len(kp) - len(unique_kps)} duplicate keypoints.')
            
        return unique_kps
    
    def get_des(self, kp: List[cv2.KeyPoint]):
        des = []
        
        for p in kp:
            desc = np.zeros((NUM_REGIONS + 2, NUM_REGIONS + 2, NUM_BINS)) # 2 was added so that we can deal with the borders more easily
            
            octave = self.octaves[p.octave]
            scale = 2 ** -p.octave
            img, mag, angle = octave.img, octave.mag, octave.angle
            
            sub_width = 3 * 0.5 * p.size * scale
            radius = np.round(sub_width * (NUM_REGIONS + 1) * math.sqrt(2) * 0.5).astype(int)
            
            x, y = int(p.pt[0] * scale), int(p.pt[1] * scale)
            orient = p.angle
            
            h, w = img.shape
            
            sin_orient = math.sin(-orient / 180 * math.pi)
            cos_orient = math.cos(-orient / 180 * math.pi)    
            for yy in range(-radius, radius + 1):
                for xx in range(-radius, radius + 1):
                    y_rot = xx * sin_orient + yy * cos_orient
                    x_rot = xx * cos_orient - yy * sin_orient
                    
                    bin_y = y_rot / sub_width + 1.5
                    bin_x = x_rot / sub_width + 1.5
                    
                    if bin_x <= -1 or bin_x >= 4 or bin_y <= -1 or bin_y >= 4:
                        continue
                    
                    y_img = y + yy
                    x_img = x + xx
                    
                    if x_img < 1 or x_img >= w - 1 or y_img < 1 or y_img >= h - 1:
                        continue
                    
                    m = self.get_val_by_interpolation(mag, x_img, y_img)
                    a = self.get_val_by_interpolation(angle, x_img, y_img)
                    
                    weight = m * self.gaussian_mask(0.75, y_rot / sub_width, x_rot / sub_width) # 3 * sigma = 2.25

                    bin_angle = ((a - p.angle) / 360 * NUM_BINS) % NUM_BINS
                    
                    # Reference: https://github.com/rmislam/PythonSIFT
                    bin_y_floor, bin_x_floor, bin_angle_floor = np.floor([bin_y, bin_x, bin_angle]).astype(int)
                    row_fraction, col_fraction, orientation_fraction = bin_y - bin_y_floor, bin_x - bin_x_floor, bin_angle - bin_angle_floor
                    bin_angle_floor = bin_angle_floor % NUM_BINS

                    c1 = weight * row_fraction
                    c0 = weight * (1 - row_fraction)
                    c11 = c1 * col_fraction
                    c10 = c1 * (1 - col_fraction)
                    c01 = c0 * col_fraction
                    c00 = c0 * (1 - col_fraction)
                    c111 = c11 * orientation_fraction
                    c110 = c11 * (1 - orientation_fraction)
                    c101 = c10 * orientation_fraction
                    c100 = c10 * (1 - orientation_fraction)
                    c011 = c01 * orientation_fraction
                    c010 = c01 * (1 - orientation_fraction)
                    c001 = c00 * orientation_fraction
                    c000 = c00 * (1 - orientation_fraction)

                    desc[bin_y_floor + 1, bin_x_floor + 1, bin_angle_floor] += c000
                    desc[bin_y_floor + 1, bin_x_floor + 1, (bin_angle_floor + 1) % 8] += c001
                    desc[bin_y_floor + 1, bin_x_floor + 2, bin_angle_floor] += c010
                    desc[bin_y_floor + 1, bin_x_floor + 2, (bin_angle_floor + 1) % 8] += c011
                    desc[bin_y_floor + 2, bin_x_floor + 1, bin_angle_floor] += c100
                    desc[bin_y_floor + 2, bin_x_floor + 1, (bin_angle_floor + 1) % 8] += c101
                    desc[bin_y_floor + 2, bin_x_floor + 2, bin_angle_floor] += c110
                    desc[bin_y_floor + 2, bin_x_floor + 2, (bin_angle_floor + 1) % 8] += c111

            desc = desc[1:-1, 1:-1, :]
            desc = desc.flatten()
            threshold = 0.2 * np.linalg.norm(desc)
            desc[desc > threshold] = threshold
            desc /= (np.linalg.norm(desc) + 1e-8)

            des.append(desc)
        
        return np.array(des, dtype=np.float32)
    
    # main function
    def detectAndCompute(self, base_img: cv2.typing.MatLike, mask: None = None):
        self.__init__()
        logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
        
        PRINT = base_img.shape[0] == 548
        
        self.octaves.append(Octave(base_img))
        logging.info(f'Initialized SIFT class with base image of size {base_img.shape}.')
        
        num_octaves = int(round(math.log(min(base_img.shape)) / math.log(2) - 4))
        for i in range(num_octaves - 1):
            last_img = self.octaves[i].img
            img = cv2.resize(last_img, (int(last_img.shape[1] / 2), int(last_img.shape[0] / 2))) # add Gaussian blur 1.6?
            self.octaves.append(Octave(img))
            if PRINT:
                debug.save_doc_image(img, f"octave-{i+1}")
            logging.info(f'Added octave No.{i + 1} from image of size {img.shape}.')
        
        all_kps = []
        
        for octave_index, octave in enumerate(self.octaves):
            img = octave.img
            kps: List[cv2.KeyPoint] = self.get_kp(img)
            if PRINT:
                debug.save_doc_image(cv2.drawKeypoints(img, kps, img, flags=2), f"keypoint-octave-{octave_index}")
            logging.info(f'Detected {len(kps)} keypoints from octave No.{octave_index}.')
            for kp in kps:
                kp.size = 2 ** (octave_index + 1)
                kp.pt = tuple(np.array(kp.pt) * (2 ** (octave_index)))
                kp.octave = octave_index
            all_kps.extend(kps)
        
        self.compute_kp_orient(all_kps)
        all_kps = self.remove_duplicate_kp(all_kps)
        
        if PRINT:
            print_kps = all_kps.copy()
            for p in print_kps:
                p.size *= 4
            debug.save_doc_image(cv2.drawKeypoints(base_img, print_kps, base_img, flags=4), f"keypoint-all")
        
        des = self.get_des(all_kps)
        
        return all_kps, des