import cv2

def bgr2rgb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def rgb2bgr(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def mirrorImage(image):
    image = cv2.flip(image, 1)
    return image