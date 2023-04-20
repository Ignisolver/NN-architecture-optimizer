import numpy as np
import cv2
import random
set_path = r'C:\Users\Lenovo\Documents\stochastyka\set'

arr = []

for photo in os.listdir(set_path):
    im = cv2.imread(set_path + '\\' + photo, 0)
    reshaped = np.reshape(im, (50*50))
    arr.append((int('class1' in photo), reshaped))

random.shuffle(arr)
data = np.array(arr)

train_set = data[:int(len(data)*3/4)]
test_set = data[int(len(data)*3/4):]

print(train_set.shape)
print(test_set.shape)