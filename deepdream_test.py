import numpy as np
import time
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.recipes.caricature import feature_inversion
import cv2

img = cv2.imread("pauldebevec.jpg").astype(np.float32)/255.0
img = cv2.resize(img,(256,256))

model = models.InceptionV1()
model.load_graphdef()

start = time.time()
result = feature_inversion(img, model, "mixed4e",verbose=False)
end = time.time()

print(end - start)
print(result.shape)

cv2.imwrite("test_0.jpg",result[0]*255)
cv2.imwrite("test_1.jpg",result[1]*255)