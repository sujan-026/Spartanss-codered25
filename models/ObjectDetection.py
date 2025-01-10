# import cv2
# from ultralytics import YOLO

# def main():
#     try:
#         # Load the YOLOv11 segmentation model
#         model = YOLO('yolo11m-seg')  # Ensure 'yolo11m-seg' is valid
        
#         # Open the external webcam
#         cap = cv2.VideoCapture(0)  # Change to "/dev/video1" for Raspberry Pi
#         if not cap.isOpened():
#             print("Error: Could not open the webcam.")
#             return

#         # Optionally set a smaller resolution and frame rate for smoother processing
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv2.CAP_PROP_FPS, 15)

#         while True:
#             # Read a frame from the webcam
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Could not read frame.")
#                 break

#             # Run YOLO segmentation inference on the frame
#             results = model(frame)

#             # Extract the annotated frame with segmentation results
#             annotated_frame = results[0].plot()  # Includes segmentation masks and bounding boxes

#             # Display the annotated frame
#             cv2.imshow("YOLOv11 Segmentation Live Feed", annotated_frame)

#             # Exit on pressing 'q'
#             if cv2.waitKey(30) & 0xFF == ord('q'):
#                 break

#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         # Release resources
#         if 'cap' in locals() and cap.isOpened():
#             cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()





import cv2
from ultralytics import YOLO

class YOLOSegmentation:
    def __init__(self):
        # Load the YOLOv11 segmentation model
        self.model = YOLO('yolo11m-seg')  # Ensure 'yolo11m-seg' is valid

    def process_frame(self, frame):
        # Run YOLO segmentation inference on the frame
        results = self.model(frame)

        # Extract the annotated frame with segmentation results
        annotated_frame = results[0].plot()  # Includes segmentation masks and bounding boxes
        return annotated_frame
