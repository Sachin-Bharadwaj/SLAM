import cv2
import numpy as np
from matplotlib import pyplot as plt
from path import Path
from einops import rearrange

orb = cv2.ORB_create(1000)
# disp = Display()

def extract_good_features(image):
    '''
    :param image: HxWxC
    :param resize_tuple: (W, H)
    :return:
    '''
    # convert to grayscale
    im_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute good features to track, <Nfeatures, 1, 2>
    features = cv2.goodFeaturesToTrack(im_grey, 3000, 0.02, 3)
    return features


def process_frame(image, resize_tuple:tuple):
    '''

    :param image: H,W,C
    :param resize_tuple: (W, H)
    :return:
    '''
    H, W = image.shape[0:2]
    im = cv2.resize(image, resize_tuple)
    # compute good features to track, it returns corner on the image
    features = extract_good_features(im)
    for f in features:
        u, v = map(lambda x: int(x), f.squeeze())
        cv2.circle(im, (u,v), color=(0,0,255), radius=3)

    # plot the detected corner features on the image
    cv2.imshow("Frame", im)
    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    fname = Path("./Data/Video1.mp4")
    cap = cv2.VideoCapture(fname)

    if cap.isOpened() == False:
        print("Error opening Video object")

    # just read one frame to find the size of image stream
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            H, W = frame.shape[0:2]
        break

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            #print(frame.shape)
            #extract_good_features(frame, (W, H//2))
            process_frame(frame, (W, H//2))
            if 0:
                cv2.imshow("Frame", frame)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
