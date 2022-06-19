import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def find_car(input_dir, output_cars='output.csv'):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    with open('image_counter.txt') as image_counter_file:
        image_count = int(image_counter_file.read())

    # Images

    output = open(output_cars, 'w')

    # Inference
    for i in tqdm(range(image_count)):
        img = f'{i+1:05d}.jpg'
        path = f'{input_dir}/{img}'

        results = model(path)
        detected_objects = np.array(results.pandas().xyxy[0]['name'])
        is_car = np.any([name in detected_objects for name in ('car', 'truck', 'bus')])
        output.write(f'{img},{is_car}\n')

    output.close()

find_car('output')