from pathlib import Path
import cv2
import numpy as np
from datetime import datetime, timezone
import json
from mcap.writer import Writer
import base64
# Set a limit for the number of segments to process
MAX_SEGMENTS_TO_PROCESS = 2  # Change this value to your desired limit
MAX_TOTAL_FRAMES_TO_PROCESS = 3  # Set your total frame limit here

# Path to the dataset data
path_to_data = Path("/home/adrian/foxglove_work/6_comma2k19/DATA")
output_mcap_file = "/home/adrian/foxglove_work/6_comma2k19/output.mcap"

def seconds_to_nanoseconds(timestamp_in_seconds):
    # Convert seconds to nanoseconds and truncate
    timestamp_in_nanoseconds = int(timestamp_in_seconds * 1_000_000_000)
    return timestamp_in_nanoseconds

def timestamp_ns_to_iso8601(timestamp_ns):
    return datetime.fromtimestamp(int(timestamp_ns) / 1e9, tz=timezone.utc).isoformat()
  
# Function to open json schemas
def open_json_schema(file_name):
  path = Path(__file__).parent / file_name
  with open(path, "r") as schema_f:
    schema = json.load(schema_f)
    return schema
  
  
  



# Directory structure of each segment
#+-- segment_number
#        |
#        +-- preview.png (first frame video)
#        +-- raw_log.bz2 (raw capnp log, can be read with openpilot-tools: logreader)
#        +-- video.hevc (video file, can be read with openpilot-tools: framereader)
#        +-- processed_log/ (processed logs as numpy arrays, see format for details)
#        +-- global_pos/ (global poses of camera as numpy arrays, see format for details)

#The poses of the camera and timestamps of every frame of the video are stored
#as follows:
#  frame_times: timestamps of video frames in boot time (s)
#  frame_gps_times: timestamps of video frames in gps_time: ([gps week (weeks), time-of-week (s)])
#  frame_positions: global positions in ECEF of camera(m)
#  frame_velocities: global velocity in ECEF of camera (m/s)
#  frame_orientations: global orientations as quaternion needed to
#                      rotate from ECEF  frame to local camera frame
#                      defined as [forward, right, down] (hamilton quaternion!!!!)
def process_segment(segment, total_frames_limit, total_frames_processed, writer, image_channel_id):
    # We search for the video
    video_file = None
    global_pos_dir = None
    frame_times = None
    
    for file in segment.iterdir():   
        if file.suffix == ".hevc":  # We check if the file is a .hevc video
            video_file = file
            
        print(file.name)
        if file.name == "global_pose":
            for subfile in file.iterdir():
                if subfile.name == "frame_times":
                    frame_times = np.load(str(subfile))

    print(frame_times.shape)    
    cap = cv2.VideoCapture(str(video_file))
    
    # Initialize frame counter for the current segment
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
  
        # Break the loop if no frame is returned (end of video)
        if not ret:
            print("Reached the end of the video.")
            break
        
        # Process the frame timestamp
        frame_timestamp_seconds = frame_times[frame_number]
        frame_timestamp_nanoseconds = seconds_to_nanoseconds(frame_timestamp_seconds)
        frame_timestamp_iso = timestamp_ns_to_iso8601(frame_timestamp_nanoseconds)
        print(frame_timestamp_iso)
        
        # Image array to png
        _, buffer = cv2.imencode('.png', frame)

        # to base 64
        img_str = base64.b64encode(buffer).decode('utf-8')
        # Schema for foxglove.CompressedImage
        image_message = {
                    "timestamp": frame_timestamp_iso,
                    "frame_id": "camera_frame",
                    "format": "png",
                    "data": img_str
                }
    
        # We add the message
        writer.add_message(
            channel_id=image_channel_id,
            log_time=frame_timestamp_nanoseconds,
            data=json.dumps(image_message).encode('utf-8'),
            publish_time=frame_timestamp_nanoseconds,
        )
        
        ## Optionally display the frame
        #cv2.imshow('Frame', frame)
        #
        ## Press 'q' to quit early
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break 
        
        # Increment frame counter
        frame_number += 1

        # Update total frame count
        total_frames_processed[0] += 1
        
        # Check if the total limit has been reached
        if total_frames_processed[0] >= total_frames_limit:
            print(f"Reached the limit of {total_frames_limit} total frames processed.")
            break





writer = Writer(output_mcap_file)
writer.start()

compressed_image_schema = open_json_schema("json_schemas/CompressedImage.json") # Schema for the topic
compressed_image_schema_id = writer.register_schema(                            # Schema id
        name="foxglove.CompressedImage",
        encoding="jsonschema",
        data=json.dumps(compressed_image_schema).encode(),
    )

image_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=compressed_image_schema_id,
    topic="rgb_camera",
    message_encoding="json",
)

with open(output_mcap_file, 'wb') as mcap_file:
  for chunk in path_to_data.iterdir():
    for route in chunk.iterdir():
        print(f"Processing route: {route.name} in chunk: {chunk.name}")
        segments = []
        
        for segment in route.iterdir():
            segments.append(segment)
        
        # Sort segments based on their numeric names
        segments.sort(key=lambda x: int(x.name))
        
        total_frames_processed = [0]  # Initialize total frames processed as a list
        
        for segment in segments:
            print(f"Segment: {segment.name}")
            process_segment(segment, MAX_TOTAL_FRAMES_TO_PROCESS, total_frames_processed, writer, image_channel_id)
            
            # Break if the total frame limit has been reached
            if total_frames_processed[0] >= MAX_TOTAL_FRAMES_TO_PROCESS:
                print(f"Reached the limit of {MAX_TOTAL_FRAMES_TO_PROCESS} total frames processed.")
                break
        
        if total_frames_processed[0] >= MAX_TOTAL_FRAMES_TO_PROCESS:
            break
writer.finish()