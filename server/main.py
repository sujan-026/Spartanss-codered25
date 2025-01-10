# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# from models import model1, model2, model3
# import uuid

# app = FastAPI()

# @app.post("/process/")
# async def process_data(model: str, file: UploadFile = File(...)):
#     # Save the input file temporarily
#     input_path = f"temp/{uuid.uuid4()}.jpg"
#     with open(input_path, "wb") as f:
#         f.write(await file.read())

#     # Select and run the model
#     if model == "model1":
#         # processed_path = model1.run(input_path)
#         processed_path = "Model 1"
#     elif model == "model2":
#         # processed_path = model2.run(input_path)
#         processed_path = "Model 2"
#     elif model == "model3":
#         # processed_path = model3.run(input_path)
#         processed_path = "Model 3"
#     else:
#         return {"error": "Invalid model name"}

#     # Return the processed image
#     return FileResponse(processed_path)





# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse

# app = FastAPI()

# @app.post("/process/")
# async def process_data(
#     model: str = Form(...),  # Form data for 'model'
#     file: UploadFile = File(...)  # Multipart file upload
# ):
#     # Log the received data
#     print(f"Model selected: {model}")
#     print(f"File received: {file.filename}")
    
#     # Respond to the client
#     return JSONResponse(content={"message": f"Model '{model}' and file '{file.filename}' received successfully."})




# import cv2
# import numpy as np
# import asyncio
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# # from models import model1, model2, model3  # Import your models

# app = FastAPI()

# @app.websocket("/video/")
# async def process_video(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection established")

#     try:
#         while True:
#             # Receive video frame and model mode
#             data = await websocket.receive_bytes()
#             frame_data, mode = data[:-1], data[-1:]  # Assuming last byte is mode

#             # Decode frame
#             frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

#             # Process frame based on mode
#             if mode == b'1':  # Model 1
#                 processed_frame = "model1"
#             elif mode == b'2':  # Model 2
#                 processed_frame = "model2"
#             elif mode == b'3':  # Model 3
#                 processed_frame = "model3"
#             else:
#                 processed_frame = frame  # No processing if mode is invalid

#             # Encode frame
#             _, buffer = cv2.imencode('.jpg', processed_frame)

#             # Send processed frame back to the client
#             await websocket.send_bytes(buffer.tobytes())

#     except WebSocketDisconnect:
#         print("WebSocket connection closed")





# import cv2
# import numpy as np
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# app = FastAPI()

# @app.websocket("/video/")
# async def show_video(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection established")

#     try:
#         while True:
#             # Receive video frame bytes from the frontend
#             data = await websocket.receive_bytes()

#             # Decode the received frame
#             frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
#             if frame is None:
#                 print("Error: Received frame is invalid")
#                 continue

#             # Display the frame in a window
#             cv2.imshow("Video Stream", frame)

#             # Break the display loop on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Quitting video stream")
#                 break

#     except WebSocketDisconnect:
#         print("WebSocket connection closed")
#     finally:
#         cv2.destroyAllWindows()
















# import cv2
# import numpy as np
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from models.yolo_segmentation import YOLOSegmentation  # Adjust path to match your structure

# app = FastAPI()

# # Initialize YOLO model
# yolo_model = YOLOSegmentation()

# @app.websocket("/video/")
# async def process_video(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection established")

#     try:
#         while True:
#             # Receive video frame bytes and mode from the client
#             data = await websocket.receive_bytes()

#             # Assume last byte of the data represents the mode
#             frame_data, mode = data[:-1], data[-1:]
            
#             # Decode the received frame
#             frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
#             if frame is None:
#                 print("Error: Received frame is invalid")
#                 continue

#             # Process the frame based on the mode
#             processed_frame = frame  # Default: no processing
#             if mode == b'1':  # Mode 1: YOLO Object Detection and Segmentation
#                 print("Processing with YOLO model")
#                 processed_frame = yolo_model.process_frame(frame)

#             # Validate and encode the processed frame
#             if processed_frame is not None and isinstance(processed_frame, np.ndarray):
#                 _, buffer = cv2.imencode('.jpg', processed_frame)
#                 await websocket.send_bytes(buffer.tobytes())
#             else:
#                 print("Error: Processed frame is invalid")

#     except WebSocketDisconnect:
#         print("WebSocket connection closed")
#     finally:
#         cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from models.ObjectDetection import YOLOSegmentation  # Adjust path to match your structure

# app = FastAPI()

# # Initialize YOLO model
# yolo_model = YOLOSegmentation()

# @app.websocket("/video/")
# async def process_video(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection established")

#     try:
#         while True:
#             # Receive video frame bytes and mode from the client
#             data = await websocket.receive_bytes()

#             # Assume last byte of the data represents the mode
#             frame_data, mode = data[:-1], data[-1:]
            
#             # Decode the received frame
#             frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
#             if frame is None:
#                 print("Error: Received frame is invalid")
#                 continue

#             # Process the frame based on the mode
#             processed_frame = frame  # Default: no processing
#             if mode == b'1':  # Mode 1: YOLO Object Detection and Segmentation
#                 print("Processing with YOLO model")
#                 processed_frame = yolo_model.process_frame(frame)

#             # Validate and encode the processed frame
#             if processed_frame is not None and isinstance(processed_frame, np.ndarray):
#                 _, buffer = cv2.imencode('.jpg', processed_frame)
#                 await websocket.send_bytes(buffer.tobytes())
#             else:
#                 print("Error: Processed frame is invalid")

#     except WebSocketDisconnect:
#         print("WebSocket connection closed")
#     finally:
#         cv2.destroyAllWindows()


# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from models.ObjectDetection import YOLOSegmentation  # First model
from models.TravelMode import  DepthDetection # Second model

app = FastAPI()

# Initialize YOLO model (Mode 1)
yolo_model = YOLOSegmentation()

# Initialize Depth Detection model (Mode 2)
depth_detection_model = DepthDetection()

@app.websocket("/video/")
async def process_video(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    try:
        while True:
            # Receive video frame bytes and mode from the client
            data = await websocket.receive_bytes()

            # Assume last byte of the data represents the mode
            frame_data, mode = data[:-1], data[-1:]

            # Decode the received frame
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Error: Received frame is invalid")
                continue

            # Process the frame based on the mode
            processed_frame = frame  # Default: no processing
            if mode == b'1':  # Mode 1: YOLO Object Detection and Segmentation
                print("Processing with YOLO model (Mode 1)")
                processed_frame = yolo_model.process_frame(frame)
            elif mode == b'2':  # Mode 2: Depth Estimation and Spatial Audio
                print("Processing with Depth Detection model (Mode 2)")
                processed_frame = depth_detection_model.process_frame(frame)

            # Validate and encode the processed frame
            if processed_frame is not None and isinstance(processed_frame, np.ndarray):
                _, buffer = cv2.imencode('.jpg', processed_frame)
                await websocket.send_bytes(buffer.tobytes())
            else:
                print("Error: Processed frame is invalid")

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        cv2.destroyAllWindows()
