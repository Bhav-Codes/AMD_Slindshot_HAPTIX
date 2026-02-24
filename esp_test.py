import serial
import time

PORT = 'COM7'      # CHANGE if Device Manager shows a different COM
BAUD = 115200

print("Opening serial port...")
ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)  # ESP32 resets when serial opens

print("Sending message...")
ser.write(b'hello esp32\n')

time.sleep(0.5)

if ser.in_waiting:
    print("From ESP32:", ser.readline().decode().strip())
else:
    print("No response from ESP32")

ser.close()
