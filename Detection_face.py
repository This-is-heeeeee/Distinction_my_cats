import cv2
import numpy
import os
import sys

cascade_filename = 'haarcascade_frontalcatface.xml'
cascade_filename_ex = 'haarcascade_frontalcatface_extended.xml'

SF = 1.05#scale factor -> 1.05, 1.3 ...
MN = 3 #minimum neighbours -> 3,4,5,6 ...

origin_path = "datas"
destination_path = "dataset"

data_class = sys.argv[1]

def videoDetector(cascade, cascade_ex):

    for root, dirs, files in os.walk("{}/{}".format(origin_path, data_class)):
        for cam_file in files:
            print(root[-1])
            cam = cv2.VideoCapture("{}/{}".format(root, cam_file))
            count = 0
            count_ex = 0
            while True:
                ret, img = cam.read()

                if img is None :
                    break

                img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                height, width, channels = img.shape
                minheight = int(height/5)
                minwidth = int(width/5)

                results = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN, minSize=(minwidth, minheight))
                #print(results)
                #print(len(results))
                results_ex = cascade_ex.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN, minSize=(minwidth, minheight))

                if count < 10:
                    for i in range(len(results)):
                        x, y, w, h = results[i]
                        crop = img[y:y + h, x:x + w].copy()
                        crop = cv2.resize(crop, dsize=(50, 50))
                        cv2.imwrite("{}/{}/{}/{}_{}-{}.jpg".format(destination_path, data_class, root[-1], cam_file[0:-4], count, i), crop)
                        print("save!")
                        if i == len(results)-1:
                            count+=1


                if count_ex < 10:
                    for i in range(len(results_ex)):
                        x, y, w, h = results_ex[i]
                        crop = img[y:y + h, x:x + w].copy()
                        crop = cv2.resize(crop, dsize=(50, 50))
                        cv2.imwrite("{}/{}/{}/{}_{}-{}.jpg".format(destination_path, data_class, root[-1], cam_file[0:-4], count_ex, i), crop)
                        print("save_ex!")
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

                #cv2.imshow('catface', img)

                if count >= 10 and count_ex >= 10 :
                    break

                if cv2.waitKey(1) > 0:
                    break

def imgDetector(cascade,cascade_ex):
    for root, dirs, files in os.walk("{}/{}".format(origin_path, data_class)):
        for img_file in files:
            img = cv2.imread("{}/{}".format(root,img_file))

            img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            height, width, channels = img.shape
            minheight = int(height / 5)
            minwidth = int(width / 5)

            results = cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN, minSize=(minwidth, minheight))
            results_ex = cascade_ex.detectMultiScale(gray, scaleFactor=SF, minNeighbors=MN, minSize=(minwidth, minheight))

            for i in range(len(results)):
                x, y, w, h = results[i]
                crop = img[y:y + h, x:x + w].copy()
                crop = cv2.resize(crop, dsize=(50, 50))
                cv2.imwrite("{}/{}/{}/{}_{}.jpg".format(destination_path, data_class, root[-1], img_file[0:-4], i), crop)

            for i in range(len(results_ex)):
                x, y, w, h = results_ex[i]
                crop = img[y:y + h, x:x + w].copy()
                crop = cv2.resize(crop, dsize=(50, 50))
                cv2.imwrite("{}/{}/{}/ex_{}_{}.jpg".format(destination_path, data_class, root[-1], img_file[0:-4], i), crop)

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
            cv2.waitKey(500)





def main():
    cascade = cv2.CascadeClassifier(cascade_filename)
    cascade_ex = cv2.CascadeClassifier(cascade_filename_ex)

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    if not os.path.exists("{}/{}".format(destination_path, data_class)):
        os.mkdir("{}/{}".format(destination_path, data_class))
    if not os.path.exists("{}/{}/0".format(destination_path, data_class)):
        os.mkdir("{}/{}/0".format(destination_path, data_class))
    if not os.path.exists("{}/{}/1".format(destination_path, data_class)):
        os.mkdir("{}/{}/1".format(destination_path, data_class))
    if not os.path.exists("{}/{}/2".format(destination_path, data_class)):
        os.mkdir("{}/{}/2".format(destination_path, data_class))

    """
    for folder in folders:
        if not os.path.exists("{}/{}".format(destination_path, folder)):
            os.mkdir("{}/{}".format(destination_path, folder))
    """

    videoDetector(cascade, cascade_ex)
    #imgDetector(cascade, cascade_ex)

if __name__ == '__main__':
    main()


