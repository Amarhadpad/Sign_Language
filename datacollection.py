import cv2
from cvzone.HandTrackingModule import HandDetector
from datetime import datetime
from pathlib import Path

from sign_pipeline import load_labels, prepare_hand_image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
LABELS = load_labels(BASE_DIR / "Model" / "labels.txt")
IMAGE_SIZE = 300
OFFSET = 20


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam. Check camera access and connection.")

    detector = HandDetector(maxHands=1)
    selected_index: int | None = None
    prepared_image = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            display_frame = frame.copy()
            prepared_image = None
            hands, display_frame = detector.findHands(display_frame)
            if hands:
                prepared_image = prepare_hand_image(
                    frame,
                    hands[0]["bbox"],
                    OFFSET,
                    IMAGE_SIZE,
                )
                if prepared_image is not None:
                    (x1, y1, x2, y2), crop, canvas = prepared_image
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Hand crop", crop)
                    cv2.imshow("Training image", canvas)

            selected_label = LABELS[selected_index] if selected_index is not None else "none"
            cv2.putText(
                display_frame,
                f"Class: {selected_label} | 1-{len(LABELS)} select | S save | Q quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Collect sign samples", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("1") <= key <= ord(str(len(LABELS))):
                selected_index = key - ord("1")
                print(f"Selected class: {LABELS[selected_index]}")
            elif key == ord("s"):
                if selected_index is None:
                    print("Select a class with its number key before saving.")
                elif prepared_image is None:
                    print("No hand detected; hold a hand in view before saving.")
                else:
                    label = LABELS[selected_index]
                    output_dir = DATA_DIR / label
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filename = datetime.now().strftime("Image_%Y%m%d_%H%M%S_%f.jpg")
                    output_path = output_dir / filename
                    if not cv2.imwrite(str(output_path), prepared_image[2]):
                        raise OSError(f"Failed to save training sample to {output_path}.")
                    print(f"Saved {label}: {output_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
