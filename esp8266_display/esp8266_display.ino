#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <MD_Parola.h>
#include <MD_MAX72xx.h>
#include <SPI.h>
#include <pgmspace.h>

// ================= MATRIX =================
#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES 4

#define DATA_PIN D7
#define CLK_PIN D5
#define CS_PIN D8

MD_Parola matrix(HARDWARE_TYPE, DATA_PIN, CLK_PIN, CS_PIN, MAX_DEVICES);

// Set these to the same Wi-Fi network as the computer running test.py.
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

ESP8266WebServer server(80);

String displayText = "";
String recognizedSign = "Waiting for a sign...";
unsigned long lastUpdate = 0;

// ================= HTML =================
const char webpage[] PROGMEM = R"=====(

<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PolySpeak Notes</title>

<style>
body { background:#020617; color:white; font-family:Arial; text-align:center; }
button { font-size:16px; padding:12px; margin:6px; width:200px; border-radius:10px; border:none; }
select { padding:10px; width:220px; }
input { padding:10px; width:70%; border-radius:8px; border:none; }
#transcript { height:250px; overflow:auto; border:1px solid white; margin:10px; padding:10px; text-align:left; }
#recognizedSign { font-size:2rem; font-weight:bold; min-height:2.5rem; color:#fbbf24; }
.sign-card { max-width:640px; margin:24px auto; padding:20px; border:1px solid #334155; border-radius:14px; background:#0f172a; }
.sign-card h3 { margin-top:0; }
#signStatus { color:#94a3b8; font-size:.9rem; }
#recognizedSign { overflow-wrap:anywhere; }
.start { background:#22c55e; }
.stop { background:#ef4444; }
.save { background:#3b82f6; }
.send { background:#f59e0b; }
</style>

</head>

<body>

<h2>🌍 PolySpeak Notes</h2>

<select id="lang">
  <option value="en-IN">English (India)</option>
  <option value="hi-IN">Hindi</option>
  <option value="mr-IN">Marathi</option>
  <option value="gu-IN">Gujarati</option>
  <option value="ta-IN">Tamil</option>
  <option value="te-IN">Telugu</option>
  <option value="kn-IN">Kannada</option>
  <option value="ml-IN">Malayalam</option>
  <option value="pa-IN">Punjabi</option>
  <option value="bn-IN">Bengali</option>
  <option value="ur-IN">Urdu</option>
  <option value="en-US">English (US)</option>
  <option value="fr-FR">French</option>
  <option value="de-DE">German</option>
  <option value="es-ES">Spanish</option>
  <option value="ja-JP">Japanese</option>
</select>

<br><br>

<button class="start" onclick="start()">Start</button>
<button class="stop" onclick="stop()">Stop</button>
<button class="save" onclick="saveNotes()">Save Notes</button>

<div id="status">Idle</div>

<section class="sign-card" aria-live="polite">
  <h3>🤟 Latest recognized sign</h3>
  <div id="recognizedSign">Waiting for a sign...</div>
  <div id="signStatus">Connecting to the sign recognition service...</div>
</section>

<div id="transcript"></div>

<hr style="margin:20px;">

<h3>✍️ Send Custom Text</h3>

<input type="text" id="customText" placeholder="Type message...">

<br><br>

<button class="send" onclick="sendCustom()">Send to Display</button>

<script>

let recognition;
let transcript="";
let lastSend=0;
let ip=window.location.origin;

async function updateRecognizedSign(){
  try{
    let response=await fetch(ip+"/recognized",{cache:"no-store"});
    if(!response.ok){
      document.getElementById("signStatus").innerText="Sign status unavailable (HTTP "+response.status+"). Upload the updated NodeMCU firmware.";
      return;
    }
    document.getElementById("recognizedSign").innerText=await response.text();
    document.getElementById("signStatus").innerText="Connected — updates when a sign is recognized.";
  }catch(error){
    document.getElementById("signStatus").innerText="Cannot reach the sign status endpoint. Check the NodeMCU firmware and Wi-Fi connection.";
  }
}

updateRecognizedSign();
setInterval(updateRecognizedSign,500);

// TRANSLATE
async function translateText(text){
  try{
    let url="https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q="+encodeURIComponent(text);
    let res=await fetch(url);
    let data=await res.json();
    return data[0][0][0];
  }catch(e){
    return text;
  }
}

// START
function start(){
  recognition = new webkitSpeechRecognition();
  recognition.lang = document.getElementById("lang").value;
  recognition.continuous = true;

  recognition.start();
  document.getElementById("status").innerText="Listening...";

  recognition.onresult = async function(event){
    let text = event.results[event.results.length-1][0].transcript;
    let translated = await translateText(text);

    transcript += "🗣 "+text+"\\n➡ "+translated+"\\n\\n";
    document.getElementById("transcript").innerText = transcript;

    if(Date.now()-lastSend>1500){
      fetch(ip+"/text?msg="+encodeURIComponent(translated));
      lastSend = Date.now();
    }
  }
}

// STOP
function stop(){
  if(recognition) recognition.stop();
  document.getElementById("status").innerText="Stopped";
}

// SAVE
function saveNotes(){
  let blob = new Blob([transcript],{type:"text/plain"});
  let a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "PolySpeak_Notes.txt";
  a.click();
}

// SEND CUSTOM TEXT
function sendCustom(){
  let text = document.getElementById("customText").value;

  if(text.trim()==="") return;

  fetch(ip+"/text?msg="+encodeURIComponent(text));

  transcript += "✍️ "+text+"\\n\\n";
  document.getElementById("transcript").innerText = transcript;

  document.getElementById("customText").value="";
}

</script>

</body>
</html>

)=====";

// ================= SETUP =================
void handleText();

void setup() {
  Serial.begin(9600);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  Serial.print("ESP8266 IP address: ");
  Serial.println(WiFi.localIP());

  matrix.begin();
  matrix.setIntensity(5);

  displayText = " INTRODUCING POLYSPEAK ";
  matrix.displayText(displayText.c_str(), PA_CENTER, 80, 2000, PA_SCROLL_LEFT, PA_SCROLL_LEFT);

  server.on("/", []() {
    server.send_P(200, "text/html", webpage);
  });

  server.on("/text", handleText);
  server.on("/recognized", []() {
    server.sendHeader("Cache-Control", "no-store");
    server.send(200, "text/plain; charset=utf-8", recognizedSign);
  });

  server.begin();
}

// ================= LOOP =================
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(ssid, password);
  }

  server.handleClient();

  if (matrix.displayAnimate()) {
    matrix.displayReset();
  }
}

// ================= RECEIVE =================
void handleText() {
  if (millis() - lastUpdate < 1500) {
    server.send(200, "text/plain", "SKIP");
    return;
  }

  lastUpdate = millis();

  if (server.hasArg("msg")) {
    if (server.arg("source") == "sign") {
      recognizedSign = server.arg("msg");
    }
    displayText = " " + server.arg("msg") + " ";
    displayText.toUpperCase();

    matrix.displayClear();
    matrix.displayText(displayText.c_str(), PA_CENTER, 100, 1000, PA_SCROLL_LEFT, PA_SCROLL_LEFT);
  }

  server.send(200, "text/plain", "OK");
}
