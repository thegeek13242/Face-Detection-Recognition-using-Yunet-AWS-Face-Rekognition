import os
import cv2
import random
import glob
from pprint import pprint
import json
import time
import face_collections as fcol
import threading


COL_NAME: str = 'chef-faces'  # Name of the collection

prevNumFace = 0
currNumOfFace = 0

lock = threading.Lock()


def dict_to_binary(the_dict):
    str = json.dumps(the_dict)
    binary = ' '.join(format(ord(letter), 'b') for letter in str)
    return binary


def recog_thread(image):
    timeFace = time.time()

    rand_int = random.randint(0, 10**9)
    file_path = "temp/temp" + str(rand_int) + ".jpg"

    cv2.imwrite(file_path, image)
    height, width, _ = image.shape
    resp = fcol.detect_face(file_path)

    for face in resp['FaceDetails']:
        crop_img = image[int(face['BoundingBox']['Top']*height):int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height), int(
            face['BoundingBox']['Left']*width):int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width)]

        write_path = "temp/crop" + str(rand_int) + ".jpg"
        if crop_img.size != 0:
            cv2.imwrite(write_path, crop_img)
        else:
            continue

        resp = fcol.find_face(COL_NAME, write_path)
        dictJSON = {}
        if resp is None:
            # print("No Face Found")
            pass
        elif len(resp['FaceMatches']) != 0:
            dictJSON["name"] = resp['FaceMatches'][0]['Face']['ExternalImageId']
            dictJSON["boundingBox"] = face['BoundingBox']
            dictJSON["time"] = timeFace

            with lock, open("facelogs.json", "r") as f:
                data = json.load(f)
                if (len(data) == 0):
                    data.append(dictJSON)
                elif (data[-1]["name"] != dictJSON["name"] or (timeFace-data[-1]["time"] > 2)):
                    data.append(dictJSON)

            with lock, open("facelogs.json", "w") as f:
                json.dump(data, f)
                pprint(data[-1]["name"])
        else:
            # print("No Face Found")
            pass
        

def del_temp():
    ls = glob.glob("temp/*")
    for l in ls:
        if os.path.isfile(l):
            try:
                os.remove(l)
            except:
                pass


if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    # load faces.json file
    dictJSON = {}
    with open("facelogs.json", "w") as f:
        f.write("[\n]")

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        exit()

    weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    threads = []
    prev_time = time.time()
    while True:
        t_recog = threading.Thread(target=lambda: recog_thread(image))
        threads.append(t_recog)
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []
        currNumOfFace = len(faces)
        if currNumOfFace != prevNumFace:
            # print("Number of faces: ", currNumOfFace)
            if not t_recog.is_alive():
                t_recog.start()
            prevNumFace = currNumOfFace

        for face in faces:
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font,
                        scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

        curr_time = time.time()
        # print(f"Curr time: {curr_time}, Prev time: {prev_time}")
        if curr_time > prev_time + 60:
            for t in threads:
                if t.is_alive():
                    t.join()
            del_temp()
            prev_time = curr_time

    cv2.destroyAllWindows()
