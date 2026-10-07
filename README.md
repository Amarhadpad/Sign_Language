# Sign language converter

Recognizes a small set of hand signs from a webcam using the model in `Model/`.

## Setup on Windows

Install Python 3.11, then run these commands from the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test.py
```

Allow camera access when prompted. Stop the app with `Ctrl+C` in the terminal.

The app requires a working webcam. The trained model and its labels are in `Model/`.
By default it loads `Model/keras_model.h5`. To test a candidate model without
replacing the current one, set `SIGN_MODEL_PATH` to the candidate file before
starting the app.

## Deploy a browser app with Streamlit Community Cloud

The repository also includes `app.py`, which supports taking a webcam snapshot
or uploading a photo for single-image sign recognition. It does not stream
continuous video. The app uses the checked-in model and labels in `Model/`.

1. Push this project to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with
   GitHub.
3. Select **Create app**, choose the `Amarhadpad/Sign_Language` repository and
   the `main` branch, and set the app file path to `app.py`.
4. In **Advanced settings**, select Python 3.11, then deploy.

Streamlit Cloud installs the Python packages in `requirements.txt` and Linux
system packages in `packages.txt`. The Streamlit and protobuf versions are
bounded to remain compatible with TensorFlow and MediaPipe. Captured training
images and locally trained candidate models are not included in the
repository; add only a reviewed model file if you intend to deploy a different
model.

## Collect samples and retrain the seven signs

The project does not include the original training images. Collect new samples
with the webcam before retraining:

```powershell
.\.venv\Scripts\python.exe datacollection.py
```

In the camera window, press `1` through `7` to select the matching sign in
`Model/labels.txt`, make that hand sign, and press `s` to save one sample.
Change your hand position slightly between captures. Collect at least 100
varied samples for each sign, including different backgrounds, lighting,
distances, and hand positions. Press `q` to finish. Samples are saved under
`Data/<sign>/`.

Train and validate a candidate model with:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Training uses a held-out validation split from every sign and starts from the
current model. It first fine-tunes the classifier head, then the final
feature-extractor layers, with training-only image augmentation. The candidate
is saved as `Model/keras_model_candidate.h5`; the current
`Model/keras_model.h5` is not overwritten. Compare the printed validation
accuracy with the original before choosing whether to replace the current
model. Validation results are more useful when samples vary across collection
sessions rather than being nearly identical consecutive frames.

## Send recognized signs to an ESP8266 display

The Python app can send each stable recognized sign to an ESP8266 running the
`esp8266_display/esp8266_display.ino` web-server sketch. Open that sketch in
the Arduino IDE, install the ESP8266 board support and the `MD_Parola` library,
and replace `YOUR_WIFI_SSID` and `YOUR_WIFI_PASSWORD` with your Wi-Fi details.
Connect the computer and ESP8266 to the same Wi-Fi network. The board prints
its IP address to the Arduino Serial Monitor at 9600 baud after connecting.
Open `http://<ESP8266-IP>/` in a browser on the same Wi-Fi network. The page
shows the latest sign under “Recognized sign” and sends that same text to the
LED matrix when the Python app recognizes it.

In PowerShell, set the board's IP and start the app:

```powershell
$env:SIGN_DISPLAY_URL = "http://192.168.1.50"
.\.venv\Scripts\python.exe test.py
```

Replace `192.168.1.50` with the ESP8266's actual IP address. Leave
`SIGN_DISPLAY_URL` unset to run recognition without a display. The app sends a
sign after it remains recognized briefly, and spaces requests to respect the
sketch's 1.5-second rate limit. Press `q` in the camera window to quit.

Change the Wi-Fi password in the sketch before using it, since credentials
should not be shared or committed to source control.

## Simulate the display without the NodeMCU

Run the local display simulator in a first PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe display_simulator.py
```

Open `http://127.0.0.1:8765/` in a browser. In a second terminal, start sign
recognition with the simulator as the display:

```powershell
$env:SIGN_DISPLAY_URL = "http://127.0.0.1:8765"
.\.venv\Scripts\python.exe test.py
```

Recognized signs appear on the simulator page and in its scrolling matrix
preview. Stop each program with `Ctrl+C` in its terminal, or close the webcam
window with `q`.
