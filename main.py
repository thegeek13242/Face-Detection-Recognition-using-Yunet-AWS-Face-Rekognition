import os
import numpy as np
import cv2
from pprint import pprint
import face_collections as fcol
import threading


COL_NAME: str = 'chef-faces'  # Name of the collection
prevNumFace = 0
currNumOfFace = 0

def recog_thread():
    
    print("API Called")
    cv2.imwrite("temp.jpg", image)
    resp = fcol.detect_face("temp.jpg")
    for face in resp['FaceDetails']:
        crop_img = image[int(face['BoundingBox']['Top']*height):int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height), int(
            face['BoundingBox']['Left']*width):int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width)]
        
        cv2.imwrite("crop.jpg", crop_img)
        
        resp = fcol.find_face(COL_NAME, "crop.jpg")
        if resp is None:
            print("No Face Found")
            return
        pprint(resp['FaceMatches'][0]['Face']['ExternalImageId'])

    
if __name__ == "__main__":

    directory = os.path.dirname(__file__)
    # capture = cv2.VideoCapture(os.path.join(directory, "image.jpg"))
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        exit()

    weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    
    while True:
        t_recog = threading.Thread(target=recog_thread)

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
            print("Number of faces: ", currNumOfFace)
            if not t_recog.is_alive():
                t_recog.start()
        #     print("API Called")
        #     cv2.imwrite("temp.jpg", image)
        #     resp = fcol.detect_face("temp.jpg")
        #     for face in resp['FaceDetails']:
        #         crop_img = image[int(face['BoundingBox']['Top']*height):int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height), int(
        #             face['BoundingBox']['Left']*width):int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width)]
                
        #         cv2.imwrite("crop.jpg", crop_img)
        #         try:
        #             resp = fcol.find_face(COL_NAME, "crop.jpg")
        #         except:                                             # TODO: Add Specific Error
        #             print("No face found")
        #             continue
        #         pprint(resp)
            prevNumFace = currNumOfFace

        for face in faces:
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # landmarks = list(map(int, face[4:len(face)-1]))
            # landmarks = np.array_split(landmarks, len(landmarks)/2)
            # for landmark in landmarks:
            #     radius = 5
            #     thickness = -1
            #     cv2.circle(image, landmark, radius,
            #                color, thickness, cv2.LINE_AA)

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

    cv2.destroyAllWindows()


