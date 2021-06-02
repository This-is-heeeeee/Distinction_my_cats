import cv2
import numpy
import os

cascade_filename = 'haarcascade_frontalcatface.xml'
cascade_filename_ex = 'haarcascade_frontalcatface_extended.xml'

SF = 1.05 #scale factor -> 1.05, 1.3 ...
MN = 3 #minimum neighbours -> 3,4,5,6 ...

origin_path = "data"
destination_path = "dataset"
folders = ["img", "cam"]

def videoDetector(cascade, cascade_ex):
    fname = ''

    for root, dirs, files in os.walk("{}/{}".format(origin_path, folders[1])):
        #print(files)
        for cam_file in files:
            fname = cam_file
            cam = cv2.VideoCapture("{}/{}".format(root, cam_file))
            count = 0
            count_ex = 0
            while True:
                ret, img = cam.read()
                img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                results = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)
                print(results)
                print(len(results))
                results_ex = cascade_ex.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)

                if count < 10:
                    for i in range(len(results)):
                        x, y, w, h = results[i]
                        crop = img[y:y + h, x:x + w].copy()
                        cv2.imwrite("{}/{}/{}_{}-{}.jpg".format(destination_path, folders[1], cam_file[0:-4], count, i), crop)
                        if i == len(results)-1:
                            count+=1


                if count_ex < 10:
                    for i in range(len(results_ex)):
                        x, y, w, h = results_ex[i]
                        crop = img[y:y + h, x:x + w].copy()
                        cv2.imwrite("{}/{}/{}_{}-{}.jpg".format(destination_path, folders[1], cam_file[0:-4], count_ex, i), crop)
                        if i == len(results_ex)-1:
                            count_ex+=1

                """
                if len(results) > 0 and count < 10:
                    count = count + 1
                    for box in results:
                        x, y, w, h = box
                        crop = img[y:y+h, x:x+w].copy()
                        cv2.imwrite("{}/{}/{}_{}.jpg".format(destination_path, folders[1], cam_file[0:-4], count), crop)
                        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                if len(results_ex) > 0 and count_ex < 10:
                    count_ex = count_ex + 1
                    for box in results_ex:
                        x, y, w, h = box
                        crop = img[y:y+h, x:x+w].copy()
                        cv2.imwrite("{}/{}/ex_{}_{}.jpg".format(destination_path, folders[1], cam_file[0:-4], count_ex), crop)
                        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                """

                cv2.imshow('catface', img)

                if count >= 10 and count_ex >= 10 :
                    break

                if cv2.waitKey(1) > 0:
                    break

def imgDetector(cascade,cascade_ex):
    for root, dirs, files in os.walk("{}/{}".format(origin_path, folders[0])):
        for img_file in files:
            img = cv2.imread("{}/{}".format(root,img_file))

            img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)
            results_ex = cascade_ex.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN)

            for i in range(len(results)):
                x, y, w, h = results[i]
                crop = img[y:y + h, x:x + w].copy()
                cv2.imwrite("{}/{}/{}_{}.jpg".format(destination_path, folders[0], img_file[0:-4], i), crop)

            for i in range(len(results_ex)):
                x, y, w, h = results_ex[i]
                crop = img[y:y + h, x:x + w].copy()
                cv2.imwrite("{}/{}/ex_{}_{}.jpg".format(destination_path, folders[0], img_file[0:-4], i), crop)

            """
            if len(results) > 0:
                for box in results:
                    x, y, w, h = box
                    crop = img[y:y + h, x:x + w].copy()
                    cv2.imwrite("{}/{}/{}".format(destination_path, folders[0], img_file), crop)
                    #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

            if len(results_ex) > 0:
                for box in results_ex:
                    x, y, w, h = box
                    crop = img[y:y + h, x:x + w].copy()
                    cv2.imwrite("{}/{}/ex_{}".format(destination_path, folders[0], img_file), crop)
                    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            """

            cv2.imshow('catface', img)
            #cv2.imwrite("{}/{}/{}".format(destination_path, folders[0], img_file), img)
            cv2.waitKey(5000)





def main():
    cascade = cv2.CascadeClassifier(cascade_filename)
    cascade_ex = cv2.CascadeClassifier(cascade_filename_ex)

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    for folder in folders:
        if not os.path.exists("{}/{}".format(destination_path, folder)):
            os.mkdir("{}/{}".format(destination_path, folder))

    videoDetector(cascade, cascade_ex)
    imgDetector(cascade, cascade_ex)

if __name__ == '__main__':
    main()


