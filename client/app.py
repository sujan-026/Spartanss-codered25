import cv2
import asyncio
import websockets
import numpy as np
async def stream_video(mode):
    uri = "ws://10.250.57.12:8000/video/"  # Replace <backend-ip> with backend server IP
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # Open the webcam

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from webcam")
                break

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)

            # Append mode as the last byte
            frame_data = buffer.tobytes() + mode

            # Send the frame to the backend
            await websocket.send(frame_data)

            # Receive the processed frame from the backend
            processed_frame_data = await websocket.recv()

            # Decode and display the processed frame
            processed_frame = cv2.imdecode(
                np.frombuffer(processed_frame_data, np.uint8), cv2.IMREAD_COLOR
            )
            cv2.imshow("Processed Video Stream", processed_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video stream")
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the client with dynamic mode switching (example: start with YOLO)
asyncio.run(stream_video(b'1'))