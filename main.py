from tensorflow.keras.utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import threading
import time
import requests
import json
# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model("models/fer2013_mini_XCEPTION.102-0.66.hdf5")
# lb = pickle.loads(open(args["label_bin"], "rb").read())
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]


emotions_data = {
    "angry": 0,
    "disgust": 0,
    "scared":  0,
    "happy": 0,
    "sad": 0,
    "surprised": 0,
    "neutral": 0
}
previous_state = {
    "angry": 0,
    "disgust": 0,
    "scared":  0,
    "happy": 0,
    "sad": 0,
    "surprised": 0,
    "neutral": 0
}

currentMood = ""


def get_emotions():
    global previous_state
    while True:
        global currentMood
        global previous_state
        time.sleep(5)
        newMood = get_currentEmotion(
            previous_state.copy(), emotions_data.copy())
        previous_state = emotions_data.copy()
        if newMood != currentMood and newMood in ["sad", "happy", "angry", "neutral"]:
            # global currentMood
            currentMood = newMood
            resp = requests.post(
                'https://scaredgrippingcalculators.ghelanibhavin.repl.co/play', json={"mood": newMood})
            playlistname = json.loads(resp.text)
            print(
                f'playing https://open.spotify.com/playlist/{playlistname["playlist"]} for {currentMood}')


def get_currentEmotion(previousState, currentState):
    current = {'emotion': 0, "key": "neutral"}
    # print(currentState)
    data = {
        "angry": currentState["angry"] - previousState["angry"],
        "disgust": currentState["disgust"] - previousState["disgust"],
        "scared":   currentState["scared"] - previousState["scared"],
        "happy": currentState["happy"] - previousState["happy"],
        "sad": currentState["sad"] - previousState["sad"],
        "surprised": currentState["surprised"] - previousState["surprised"],
        "neutral": currentState["neutral"] - previousState["neutral"]
    }
    # print(data)
    for key, value in data.items():
        if value > current['emotion']:
            current['emotion'] = value
            current['key'] = key

    return current['key']


t1 = threading.Thread(target=get_emotions)
t1.start()

#feelings_faces = []
# for index, emotion in enumerate(EMOTIONS):
# feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    # reading the frame
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        emotions_data[label] += 1

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            #  label
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi, verbose=0)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        emotions_data[label] += 1
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
    else:
        continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)

        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
