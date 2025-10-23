
import asyncio
import json
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from av import VideoFrame
from PIL import Image

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to('cuda' if torch.cuda.is_available() else 'cpu')
pcs = set()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        width, height = image.size

        text_queries = [["a television remote control", "tv remote", "remote control"]]
        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([(height, width)], device=model.device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.3,
            text_labels=text_queries
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["text_labels"]

        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            cv2.putText(img, f"{label}:{score:.2f}", (xmin, max(ymin-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            local_video = VideoTransformTrack(track)
            pc.addTrack(local_video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.Response(content_type="application/json", text=json.dumps({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }))

app = web.Application()
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    web.run_app(app, port=8080)
