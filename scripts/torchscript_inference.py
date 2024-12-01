import argparse
import cv2
import time
import torch
import numpy as np

class TorchScriptYOLOv8Inference:
    def __init__(self, model_path, video_path, output_path, device="cuda:0"):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model = torch.jit.load(self.model_path)
        if 'cuda' in self.device and torch.cuda.is_available():
            model = model.to(self.device)
            print(f"Using {self.device} for inference")
        else:
            model = model.to('cpu')
            print("Using CPU for inference")
        return model

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_processing_time = 0
        fps_sum = 0

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_frame = cv2.resize(input_frame, (128,128))  # YOLOv8 expects 640x640 input
            input_tensor = torch.from_numpy(input_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(self.device)

            # Perform inference
            with torch.no_grad():
                results = self.model(input_tensor).cpu()
            # print(results)
            output = np.squeeze(results).cpu().numpy().astype(dtype=np.float32)
            class_value=np.argmax(output)
            print('class:', class_value, ' scores:', np.max(output))
            if class_value ==1:
                mean_rgb = frame.mean(axis=0).mean(axis=0).astype(int)
                for c in range(3):
                    frame[:,:,c]=mean_rgb[c]
            # # Assuming results contains the detection data
            # if isinstance(results, torch.Tensor):
            #     # Parse results if it's a single tensor output
            #     boxes = results[0][:, :4]  # Bounding boxes (x1, y1, x2, y2)
            #     print(boxes)
            #     scores = results[0][:, 4]  # Objectness scores
            #     print(scores)
            #     class_ids = results[0][:, 5].long()  # Class indices
            #     print(class_ids)

            # Iterate through the predictions
            # for i in range(len(boxes)):
            #     if scores[i] > 0.3:  # Confidence threshold
            #         x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            #         label = class_ids[i].item()
            #         class_label = self.class_labels[label] if label < len(self.class_labels) else "Unknown"

            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(frame, f"{class_label}: {scores[i]:.2f}", (x1, y1 - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            end_time = time.time()
            total_processing_time += end_time - start_time
            fps = 1 / (end_time - start_time)
            fps_sum += fps
            frame_count += 1

            # Display FPS on frame
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)

        print(f"Processed {frame_count} frames.")
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Average FPS: {fps_sum / frame_count:.2f}" if frame_count > 0 else "No frames processed.")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchScript YOLOv8 Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TorchScript YOLOv8 model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference ('cpu', 'cuda:0', etc.)")
    args = parser.parse_args()

    yolo_inference = TorchScriptYOLOv8Inference(
        args.model_path, args.video_path, args.output_path, args.device
    )

    yolo_inference.process_video()



