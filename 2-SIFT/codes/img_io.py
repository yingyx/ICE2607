import cv2

def load_dataset_and_target(
    dataset_path: str = 'images/dataset/',
    dataset_names: list[str] = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg'],
    target_path: str = 'images/target.jpg',
    color: int = cv2.COLOR_BGR2GRAY) -> tuple[list[cv2.typing.MatLike], cv2.typing.MatLike]:
    
    dataset = [cv2.imread(dataset_path + name) for name in dataset_names]
    target = cv2.imread(target_path)
    
    dataset = [cv2.cvtColor(img, color) for img in dataset]
    target = cv2.cvtColor(target, color)
    
    return dataset, target

def save_output(
    imgs: list[cv2.typing.MatLike],
    output_path: str = 'output/',
    suffix: str = None):
    
    for _, img in enumerate(imgs):
        path = output_path + f"{_ + 1}"
        if suffix:
            path += '-' + suffix
        path += '.png'
        
        cv2.imwrite(path, img)
        
def save_single_image(
    img: cv2.typing.MatLike,
    name: str,
    path: str
):
    
    if not name or not path:
        return
    
    cv2.imwrite(path + name + '.png', img)