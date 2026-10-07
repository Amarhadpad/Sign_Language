import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from sign_pipeline import load_labels, prepare_hand_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("SIGN_MODEL_PATH", str(BASE_DIR / "Model" / "keras_model.h5"))
)
LABELS_PATH = BASE_DIR / "Model" / "labels.txt"
DISPLAY_URL = os.environ.get("SIGN_DISPLAY_URL", "").strip().rstrip("/")
STABLE_SECONDS = 0.75
DISPLAY_TIMEOUT = 1.5
DISPLAY_INTERVAL = 1.5
NO_HAND_RESET_SECONDS = 0.5

classifier = Classifier(MODEL_PATH, LABELS_PATH)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
labels = load_labels(LABELS_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.release()
    raise RuntimeError("Could not open webcam. Check camera access and connection.")

candidate_label = None
candidate_since = 0.0
last_sent_label = None
last_hand_seen = 0.0
last_display_attempt = 0.0

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        now = time.monotonic()

        if hands:
            last_hand_seen = now
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            height, width = img.shape[:2]
            prepared = prepare_hand_image(img, hand["bbox"], offset, imgSize)

            if prepared is not None:
                (x1, y1, x2, y2), imgCrop, imgWhite = prepared

                _, index = classifier.getPrediction(imgWhite, draw=False)
                label = labels[index]

                if label != candidate_label:
                    candidate_label = label
                    candidate_since = now

                cv2.rectangle(imgOutput, (x1, max(0, y1 - 70)), (min(width, x1 + 400), y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, label, (x, max(30, y - 30)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

                if (
                    DISPLAY_URL
                    and label != last_sent_label
                    and now - candidate_since >= STABLE_SECONDS
                    and now - last_display_attempt >= DISPLAY_INTERVAL
                ):
                    last_display_attempt = now
                    try:
                        query = urlencode({"msg": label, "source": "sign"})
                        with urlopen(
                            f"{DISPLAY_URL}/text?{query}",
                            timeout=DISPLAY_TIMEOUT,
                        ) as response:
                            result = response.read().decode("utf-8", errors="replace").strip()
                            if response.status != 200 or result != "OK":
                                print(f"Display did not accept {label!r}: {result or response.status}")
                            else:
                                print(f"Sent to display: {label}")
                                last_sent_label = label
                    except (HTTPError, URLError, TimeoutError, OSError) as error:
                        print(f"Could not send sign to display: {error}")
        elif now - last_hand_seen >= NO_HAND_RESET_SECONDS:
            candidate_label = None
            last_sent_label = None

        cv2.imshow("Image", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    