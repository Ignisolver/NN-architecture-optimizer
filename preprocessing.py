import os
import shutil
import cv2

root_path = r'C:\Users\Macie\PycharmProjects\NN-architecture-optimizer'
dataset_size = 20000
directory = root_path + r'\archive'

###NOTE in dataset there is another folder outsize od the patient folders, get its path before you delete it, add it to shutil.rmtree(folder_path) and update on git
# shutil.rmtree(r'C:\Users\Macie\PycharmProjects\NN-architecture-optimizer\archive\IDC_regular_ps50_idx5')
# pruning unwanted pictures
for patient in os.listdir(directory):
    for photo in os.listdir(directory + '\\' + patient + '\\' + '0'):
        im = cv2.imread(directory + '\\' + patient + '\\' + '0' + '\\' + photo)
        if im.shape[0] != 50 or im.shape[1] != 50 or im.shape[2] != 3:
            os.remove(directory + '\\' + patient + '\\' + '0' + '\\' + photo)
    for photo in os.listdir(directory + '\\' + patient + '\\' + '1'):
        im = cv2.imread(directory + '\\' + patient + '\\' + '1' + '\\' + photo)
        if im.shape[0] != 50 or im.shape[1] != 50 or im.shape[2] != 3:
            os.remove(directory + '\\' + patient + '\\' + '1' + '\\' + photo)

# unpacking photos from patinent catalogs
directory = root_path + r'\archive'
negatives = root_path + r'\0'
positives = root_path + r'\1'

if not os.path.exists(negatives):
    os.makedirs(negatives)
if not os.path.exists(positives):
    os.makedirs(positives)

for patient in os.listdir(directory):
    for photo in os.listdir(directory + '\\' + patient + '\\' + '0'):
        shutil.move(directory + '\\' + patient + '\\' + '0' + '\\' + photo, negatives + '\\' + photo)
    for photo in os.listdir(directory + '\\' + patient + '\\' + '1'):
        shutil.move(directory + '\\' + patient + '\\' + '1' + '\\' + photo, positives + '\\' + photo)

# creating dataset
negatives_path = root_path + r'\0'
positives_path = root_path + r'\1'
set_path = root_path + r'\set'

if not os.path.exists(set_path):
    os.makedirs(set_path)

for i, photo in enumerate(os.listdir(negatives_path)):
    if i >= dataset_size:
        break
    curr_path = negatives_path + '\\' + photo
    img = cv2.imread(curr_path, cv2.IMREAD_COLOR)
    # equ = cv2.equalizeHist(img)
    cv2.imwrite(set_path + '\\' + photo, img)

for i, photo in enumerate(os.listdir(positives_path)):
    if i >= dataset_size:
        break
    curr_path = positives_path + '\\' + photo
    img = cv2.imread(curr_path, cv2.IMREAD_COLOR)
    # equ = cv2.equalizeHist(img)
    cv2.imwrite(set_path + '\\' + photo, img)