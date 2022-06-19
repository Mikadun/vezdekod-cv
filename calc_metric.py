import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def calc_metric(image, x, y, w, h):

    # prepare image
    img = cv2.imread(image)
    img = img[y:y+h, x:x+w]
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = data.reshape((data.shape[0] * data.shape[1], 3))

    # use kmeans algorithm to find color clusters
    kmeans = KMeans(n_clusters=20)
    kmeans.fit(data)

    # find color that belongs to biggest cluster
    clusters = kmeans.cluster_centers_.astype(np.int8)
    counter = Counter(kmeans.labels_)
    result = tuple(clusters[counter.most_common(1)[0][0]])

    # show image with color
    clr = np.zeros((img.shape[0], int(1 + img.shape[1] * 0.1), 3), dtype=np.uint8)
    clr[:] = result
    clr = cv2.cvtColor(clr, cv2.COLOR_RGB2BGR)

    cv2.imshow('Image with color', cv2.hconcat((img, clr)))
    cv2.waitKey(0)
    
    return result
    

if __name__ == '__main__':
    calc_metric('output/00074.jpg', 100, 150, 600, 200)