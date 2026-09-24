#include <Arduino.h>
#include <Wire.h>

void scanOnPins(int sda, int scl) {
  Serial.print("Scanning SDA=");
  Serial.print(sda);
  Serial.print(" SCL=");
  Serial.println(scl);

  Wire.end();
  // Force pull-ups before init
  pinMode(sda, INPUT_PULLUP);
  pinMode(scl, INPUT_PULLUP);
  Wire.begin(sda, scl);
  Wire.setClock(100000); // Slow clock for reliability

  int found = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    byte err = Wire.endTransmission();
    if (err == 0) {
      Serial.print("  >>> Device found at 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  Nothing found.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
}

void loop() {
  Serial.println("=============================");
  scanOnPins(8, 9);   // Default SuperMini guess
  delay(500);
  scanOnPins(6, 7);   // Alternative common pins
  delay(500);
  scanOnPins(4, 5);   // Another alternative
  delay(500);
  scanOnPins(0, 1);   // Another alternative
  Serial.println("=============================");
  delay(4000);
}