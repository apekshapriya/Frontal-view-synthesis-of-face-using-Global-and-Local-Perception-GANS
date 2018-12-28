import cv2
import numpy as np
import glob
import os

def create_dataset():
    """
    creates data the above model. Data is created by taking profile images and front images and thier facial landmark images
    :return:
    """
    batch_size = 4

    noise_input = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

    front_images = glob.glob("../../processed_data/front_images/*")
    profile_images = glob.glob("../../processed_data/profile_images/*")
    facial_landmarks = "../../processed_data/cropped_images"
    in_prof_img, in_prof_leye, in_prof_reye, in_prof_nose, in_prof_mouth, in_prof_noise,in_front_img = [],[],[],[],[],[],[]
    y = np.ones([2 * batch_size, 1])
    y[batch_size:, :] = 0
    for prof_img in profile_images:
        image_name = prof_img.split("/")[-1].split(".")[0]


        in_prof_img.append(cv2.resize(cv2.imread(prof_img), dsize=(128, 128), interpolation=cv2.INTER_LINEAR))

        leye = cv2.imread(os.path.join(facial_landmarks, "profile_images", image_name, "left_eye.jpg"))
        in_prof_leye.append(cv2.resize(leye, dsize=(40, 40), interpolation=cv2.INTER_LINEAR))

        reye = cv2.imread(os.path.join(facial_landmarks, "profile_images", image_name, "right_eye.jpg"))
        in_prof_reye.append(cv2.resize(reye, dsize=(40, 40), interpolation=cv2.INTER_LINEAR))

        nose = cv2.imread(os.path.join(facial_landmarks, "profile_images", image_name, "nose.jpg"))
        in_prof_nose.append(cv2.resize(nose, dsize=(32,40), interpolation=cv2.INTER_LINEAR))

        mouth = cv2.imread(os.path.join(facial_landmarks, "profile_images", image_name, "mouth.jpg"))
        in_prof_mouth.append(cv2.resize(mouth, dsize=(32,48), interpolation=cv2.INTER_LINEAR))

    for prof_img in front_images:

        in_front_img.append(cv2.resize(cv2.imread(prof_img), dsize=(128, 128), interpolation=cv2.INTER_LINEAR))

    x = in_front_img + in_prof_img

    return np.array(x),np.array(y),np.array(in_prof_img), np.array(in_prof_leye), np.array(in_prof_reye),\
           np.array(in_prof_nose),np.array(in_prof_mouth),np.array(in_front_img), np.array(noise_input)
