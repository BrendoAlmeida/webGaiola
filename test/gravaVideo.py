from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (640, 480)}, controls={"FrameRate": 10})
picam2.configure(config)

encoder = H264Encoder(bitrate=10000000)
output_file = "my_video.h264"

picam2.start_recording(encoder, output_file)

time.sleep(60)
picam2.stop_recording()

picam2.stop()