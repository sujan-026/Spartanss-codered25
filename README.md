# Vision AI Smart Glasses

Vision AI Smart Glasses is an assistive device designed to enhance accessibility for visually impaired individuals. The system comprises a client and a server setup to process real-time data for object detection, OCR, and audio feedback.

## Project Overview

- **Client**: Deployed on a Raspberry Pi 5 (8GB), it captures images, performs initial object detection, and runs OCR.
- **Server**: Deployed on a desktop, it processes data received from the client using advanced models and algorithms, and sends back processed results.

## Features

- **Travel Mode**: Detects objects, calculates distance and position, and announces obstacles via spatial audio feedback.
- **Read Mode**: Captures images, processes text using OCR, and reads text aloud to the user.
- **Observer Mode**: Identifies and announces multiple objects in the surroundings with spatial feedback.

## Requirements

### Hardware

- Raspberry Pi 5 (8GB model recommended)
- Camera module
- Speakers
- Desktop/laptop for server deployment
- Touch sensors connected to GPIO pins 23 (left) and 24 (right)

### Software Dependencies

All dependencies are listed in `requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

Main libraries used:
- opencv-python
- torch
- numpy
- ultralytics
- gTTS
- sounddevice
- soundfile
- pytesseract
- pyttsx3
- speechrecognition
- wavio
- huggingface_hub
- gpiozero

## Setup and Execution

### Client Setup (Raspberry Pi)

1. Clone the repository onto the Raspberry Pi:
```bash
git clone <repository-url>
cd vision-ai-smart-glasses
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Connect touch sensors to GPIO pins 23 (left touch) and 24 (right touch).

4. Run the client:
```bash
python app.py
```

### Server Setup (Desktop)

1. Clone the repository onto the desktop:
```bash
git clone <repository-url>
cd vision-ai-smart-glasses
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python main.py
```

## Workflow

### Startup
1. The client initializes on the Raspberry Pi and connects to the server.
2. The server begins listening for incoming data.

### Mode Selection
- Modes can be toggled using the right touch sensor.
- Left touch sensor provides detailed explanations for detected objects or OCR results.

### Processing
1. Images captured by the client are sent to the server for processing.
2. The server performs object detection, OCR, and other computations.
3. Processed results are sent back to the client for audio feedback.

### Exit
- Use audio prompts or stop the scripts manually to shut down the system.

## How to Run

1. Ensure the Raspberry Pi and desktop are on the same network.
2. Start the server on the desktop:
```bash
python main.py
```
3. Start the client on the Raspberry Pi:
```bash
python app.py
```
4. Follow the audio prompts to select a mode and interact with the system.

## What the Judges Need to Know

- The project enables visually impaired users to navigate, read text, and understand their surroundings using audio feedback.
- Client: Captures and sends raw data (images and text) to the server.
- Server: Processes data using models and algorithms to provide detailed feedback.
- Touch Sensors:
  - Left touch: Detailed explanation of the current context.
  - Right touch: Switch between modes.

## Future Enhancements

1. Support for multi-language OCR and text-to-speech.
2. Enhanced spatial audio processing for more immersive navigation.
3. Integration with cloud-based services for complex computations.

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Thank you for exploring Vision AI Smart Glasses! Together, let's make accessibility more innovative and inclusive.
