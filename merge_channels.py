import os
import cv2
from tqdm import tqdm

def merge_channels(input_dir: str, output_dir: str):
    IMAGE_COUNTER_PATH = f'{input_dir}/image_counter.txt'

    # read image count
    with open(IMAGE_COUNTER_PATH) as image_counter_file:
        image_count = int(image_counter_file.read())
    
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # merge channels and save each image
    for i in tqdm(range(image_count)):
        input_image_names = [f'{input_dir}/data/{i+1:05d}_{ch}.jpg' for ch in 'bgr']
        input_images = [cv2.imread(input_image_names[j], cv2.IMREAD_GRAYSCALE) for j in range(3)]

        output_image_name = f'{output_dir}/{i+1:05d}.jpg'
        output_image = cv2.merge(input_images)
        cv2.imwrite(output_image_name, output_image)


if __name__ == '__main__':
    merge_channels('.', 'output')
