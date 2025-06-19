#include <Adafruit_NeoPixel.h>

// Pin and LED setup
#define PIN 2          // Data pin connected to D2
#define NUMPIXELS 12   // Ring size

// Create NeoPixel object
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

// Store current LED states
uint32_t ledStates[NUMPIXELS];

void setup() {
  pixels.begin();
  pixels.clear();
  pixels.show();
  
  // Initialize all LEDs to off
  for(int i = 0; i < NUMPIXELS; i++) {
    ledStates[i] = 0;
  }
  
  Serial.begin(9600);
  Serial.println("READY"); // Signal to Python that Arduino is ready
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("SET:")) {
      // Parse command: SET:0,255,0,0;1,0,255,0;2,0,0,255
      // Format: SET:id,r,g,b;id,r,g,b;...
      command = command.substring(4); // Remove "SET:"
      
      // Split by semicolon for each LED command
      int lastIndex = 0;
      int nextIndex = 0;
      
      while (nextIndex != -1) {
        nextIndex = command.indexOf(';', lastIndex);
        String ledCommand;
        
        if (nextIndex == -1) {
          ledCommand = command.substring(lastIndex);
        } else {
          ledCommand = command.substring(lastIndex, nextIndex);
        }
        
        if (ledCommand.length() > 0) {
          parseLedCommand(ledCommand);
        }
        
        lastIndex = nextIndex + 1;
      }
      
      pixels.show();
      Serial.println("OK");
      
    } else if (command == "CLEAR") {
      pixels.clear();
      for(int i = 0; i < NUMPIXELS; i++) {
        ledStates[i] = 0;
      }
      pixels.show();
      Serial.println("OK");
      
    } else if (command == "STATUS") {
      // Return current LED states
      Serial.print("STATUS:");
      for(int i = 0; i < NUMPIXELS; i++) {
        uint8_t r = (ledStates[i] >> 16) & 0xFF;
        uint8_t g = (ledStates[i] >> 8) & 0xFF;
        uint8_t b = ledStates[i] & 0xFF;
        Serial.print(i);
        Serial.print(",");
        Serial.print(r);
        Serial.print(",");
        Serial.print(g);
        Serial.print(",");
        Serial.print(b);
        if (i < NUMPIXELS - 1) Serial.print(";");
      }
      Serial.println();
    }
  }
}

void parseLedCommand(String command) {
  // Parse: "id,r,g,b" or "id,,,," for no change
  int commaIndex[3];
  commaIndex[0] = command.indexOf(',');
  commaIndex[1] = command.indexOf(',', commaIndex[0] + 1);
  commaIndex[2] = command.indexOf(',', commaIndex[1] + 1);
  
  if (commaIndex[2] == -1) return; // Invalid format
  
  int id = command.substring(0, commaIndex[0]).toInt();
  String rStr = command.substring(commaIndex[0] + 1, commaIndex[1]);
  String gStr = command.substring(commaIndex[1] + 1, commaIndex[2]);
  String bStr = command.substring(commaIndex[2] + 1);
  
  // Check if ID is valid
  if (id < 0 || id >= NUMPIXELS) return;
  
  // If any color value is empty, skip this LED (keep previous state)
  if (rStr.length() == 0 || gStr.length() == 0 || bStr.length() == 0) {
    return;
  }
  
  int r = rStr.toInt();
  int g = gStr.toInt();
  int b = bStr.toInt();
  
  // Clamp values to 0-255
  r = constrain(r, 0, 255);
  g = constrain(g, 0, 255);
  b = constrain(b, 0, 255);
  
  uint32_t color = pixels.Color(r, g, b);
  pixels.setPixelColor(id, color);
  ledStates[id] = color;
}
