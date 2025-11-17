import cv2
import img_io as iio

def print_keypoint(kp: cv2.KeyPoint):
    print(f"Angle: {kp.angle}, Size: {kp.size}, Octave: {kp.octave}.")
    
def save_doc_image(img: cv2.typing.MatLike, name: str):
    iio.save_single_image(img, name, path='../doc/images/')