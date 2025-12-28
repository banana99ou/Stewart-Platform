# ESP32 Serial → Wi‑Fi Web Viewer (for Simulink)

Simulink takes the serial port, so you can’t use Serial Monitor to confirm what the ESP32 is receiving.  
This sketch keeps reading **Serial bytes from Simulink** and shows them on a simple webpage over Wi‑Fi.

## What it shows

- **Total bytes** received since boot (or since you “Clear counters”)
- **Last byte** in decimal + hex
- **Last N bytes** (up to 64) in hex
- **Last float** decoded every 4 bytes (little‑endian, as received)

## Setup

1. Open `arduinoReceive/arduinoReceive.ino` and select an **ESP32** board in Arduino IDE.
2. Flash the sketch to the ESP32.
3. Power the ESP32 and ensure it can join your Wi‑Fi:
   - SSID: `iptime`
   - Password: `Za!cW~QWdh5FC~f`
4. Find the ESP32’s IP (router client list is easiest).
5. From a device on the same LAN, open a browser to:
   - `http://<esp32-ip>/`

## Using with Simulink

- Start your Simulink model that transmits bytes/floats over the serial port.
- Open the webpage and watch **Total bytes** increase and **Last 64 bytes** change.
- If your Simulink payload is IEEE754 float packed into 4 bytes, **Last float** should also update.

## Troubleshooting

- If the page loads but shows `wifi: disconnected`, the ESP32 is not associated to the AP yet.
- If **Total bytes** stays at 0, Simulink is likely not sending (or baud mismatch).
- If bytes move but “Last float” looks wrong, your float packing or endianness may differ.


