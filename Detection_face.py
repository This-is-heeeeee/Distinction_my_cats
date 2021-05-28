import cv2
import numpy

cascade_filename = 'haarcascade_frontalcatface.xml'
cascade_filename_ex = 'haarcascade_frontalcatface_extended.xml'

SF = 1.05 #scale factor -> 1.05, 1.3 ...
MN = 3 #minimum neighbours -> 3,4,5,6 ...

def videoDetector(cam, cascade, cascade_ex):
    while True:

        ret, img = cam.read()
        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)
        results_ex = cascade_ex.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)

        for box in results:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        for box in results_ex :
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

        cv2.imshow('catface',img)
        cv2.imwrite('result_video.mp4',img)

        if cv2.waitKey(1) > 0:
            break



def imgDetector(img, cascade,cascade_ex):
    img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)
    results_ex = cascade_ex.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)

    for box in results:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
    for box in results_ex:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv2.imshow('catface',img)
    cv2.imwrite('result_image2.jpg', img)
    cv2.waitKey(10000)



def main():
    cascade = cv2.CascadeClassifier(cascade_filename)
    cascade_ex = cv2.CascadeClassifier(cascade_filename_ex)

    #cam = cv2.VideoCapture('autumn_video.mp4')
    img = cv2.imread('autumn.jpg')

    #videoDetector(cam, cascade, cascade_ex)
    imgDetector(img, cascade, cascade_ex)

if __name__ == '__main__':
    main()


