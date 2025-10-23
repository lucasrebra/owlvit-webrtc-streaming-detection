# Live Streaming Object Detection – WebRTC + OWL-ViT  
Live camera streaming via WebRTC, processed in real-time with OWL-ViT (open-vocabulary object detection).  
  
## 🔍 What this project does  
This repository provides a demo allowing you to stream camera video from one device (sender) to a server, run zero-shot object detection on the video frames using the [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) model, and return/display the processed video (with bounding boxes) to the sender or another client.  
  
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
