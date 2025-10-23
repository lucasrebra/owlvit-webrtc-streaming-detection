# Live Streaming Object Detection – WebRTC + OWL-ViT  
Live camera streaming via WebRTC, processed in real-time with OWL-ViT (open-vocabulary object detection).  
  
## 🔍 What this project does  
This repository provides a demo allowing you to stream camera video from one device (sender) to a server, run zero-shot object detection on the video frames using the [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) model, and return/display the processed video (with bounding boxes) to the sender or another client. Updated from the detector which I have made with YOLOv3 and streamlit, which was a not-really-proffesional solution for this problem as a PoC and needed to be improved. This is a best aproach, also being a proof of concept.
  
## 🧩 How it works  
1. The client (browser) captures the webcam video via WebRTC and sends it to the server.  
2. The server receives the video stream, processes each frame through the OWL-ViT model using text prompts (for example “remote control”), draws bounding boxes, then sends the processed video back to the client.  
3. The client displays the annotated video in real time.  

## 🚀 Get started  
### Prerequisites  
- Python 3.8+  
- GPU recommended for frame processing (though CPU works with reduced performance)  
- A modern browser that supports WebRTC  

### Installation  
```bash
pip install -r requirements.txt
```
### Running the server/receiver

```bash
python server/server.py
```
### Running the sender (client)

- Open client/index.html in the browser of the sender device, allow camera access, and point it to the receiver/server address.
- On another device (client/display), open the receiver display interface (if applicable) to view the annotated stream.

### Customize

- Change text_queries in server.py to detect different objects (e.g., "a cat", "a dog", "a remote control").

- Adjust the confidence threshold in post_process_grounded_object_detection() to filter detections by confidence.

- Adjust video resolution, frame rate, or skip frames to manage latency and performance.

- Use STUN/TURN servers or multiple peers if your devices are across NATs or remote networks.

### Limitations & Notes

- Latency depends on network speed, device performance, and model processing time — expect some delay compared to a standard webcam feed.

- GPU significantly improves processing speed; on CPU performance may drop or frames may lag.

- The WebRTC signalling in this demo is minimal and designed for same-network usage; production setups will need robust signalling, ICE candidates, NAT handling, etc.

- The effectiveness of zero-shot detection with OWL-ViT depends on the clarity and size of the objects, and on how well the prompts match the visual concept.
