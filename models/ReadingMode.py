# OCR , TTS WITH AI                                                                                                                                                                                                                                                         import cv2
import pytesseract
import pyttsx3
import re
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import wavio
from huggingface_hub import InferenceClient
import dotenv 

# Configuration
SAMPLE_RATE = 44100  # Sample rate in Hz
DURATION = 5  # Duration of recording in seconds
OUTPUT_FILE = "temp_audio.wav"  # Temporary file to save the recording
dotenv.load_dotenv()
HF_API_KEY = os.getenv("API_KEY")# Replace with your Hugging Face API key
MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key=HF_API_KEY)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech rate

def speak_text(text):
    """Convert text to speech using pyttsx3."""
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def record_audio():
    """Record audio using sounddevice and save it to a WAV file."""
    print("Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished. Processing...")
    wavio.write(OUTPUT_FILE, audio, SAMPLE_RATE, sampwidth=2)  # Save as WAV file

def recognize_speech():
    """Convert the recorded audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(OUTPUT_FILE) as source:
            audio_data = recognizer.record(source)  # Load the recorded audio
            print("Recognizing speech...")
            text = recognizer.recognize_google(audio_data)  # Convert speech to text
            print("Recognized Text:", text)
            return text
    except sr.UnknownValueError:
        print("Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def query_hugging_face(prompt):
    """Send the prompt to Hugging Face model and return the response."""
    try:
        print("Sending text to Hugging Face API...")
        messages = [{"role": "user", "content": prompt}]
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500,
            stream=True
        )

        response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            response += content
        print(f"Hugging Face Response:\n{response}")
        return response
    except Exception as e:
        print(f"Error communicating with Hugging Face API: {e}")
        return None

# Initialize camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'c' to capture an image and detect text. Press 'd' for speech-to-text. Press 'q' to quit.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Display the live camera feed
        cv2.imshow('Live Camera Feed', frame)

        # Wait for keypress
        key = cv2.waitKey(10) & 0xFF  # Reduced wait time for responsiveness

        if key == ord('c'):  # Capture image on pressing 'c'
            print("Image captured. Performing text detection...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_text = pytesseract.image_to_string(gray)
            filtered_text = re.sub(r'[^a-zA-Z \\n]', '', raw_text)
            if filtered_text.strip():  # If valid text is detected
                print("Filtered Text Detected:")
                print(filtered_text)
                speak_text(filtered_text)
            else:
                print("No valid alphabetic text detected.")
            print("Press 'c' to capture another image, 'd' for speech-to-text, or 'q' to quit.")

        elif key == ord('d'):  # Switch to speech-to-text mode on pressing 'd'
            record_audio()  # Record audio
            recognized_text = recognize_speech()  # Recognize speech
            if recognized_text:  # If valid text is recognized
                hf_response = query_hugging_face(recognized_text)  # Send to Hugging Face
                if hf_response:
                    print(f"Hugging Face Response:\n{hf_response}")
                    speak_text(hf_response)  # Convert Hugging Face response to speech
            print("Press 'c' to capture an image, 'd' for another speech-to-text, or 'q' to quit.")

        elif key == ord('q'):  # Exit the program on pressing 'q'
            print("Exiting program.")
            break

except KeyboardInterrupt:
    print("Program interrupted by user. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
