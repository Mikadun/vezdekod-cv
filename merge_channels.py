import os
import cv2
from tqdm import tqdm

def merge_channels(input_dir: str, output_dir: str):
    '''
    input_dir — имя директории, содержащей датасет;
    output_dir — имя директории, содержащей RGB-изображения.
    '''

    # read image count
    with open('image_counter.txt') as image_counter_file:
        image_count = int(image_counter_file.read())
    
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # merge channels and save each image
    for i in tqdm(range(image_count)):
        input_image_names = [f'{input_dir}/00{i+1:03d}_{ch}.jpg' for ch in 'bgr']
        input_images = [cv2.imread(input_image_names[j], cv2.IMREAD_GRAYSCALE) for j in range(3)]

        output_image_name = f'{output_dir}/{i+1:05d}.jpg'
        output_image = cv2.merge(input_images)
        cv2.imwrite(output_image_name, output_image)


if __name__ == '__main__':
    merge_channels('data', 'output')
