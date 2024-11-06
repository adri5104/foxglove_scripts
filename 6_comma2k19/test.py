import cv2
import torch
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime, timezone
import json
from mcap.writer import Writer
import base64
from PIL import Image
from scipy.spatial.transform import Rotation as R
from depth_anything_v2.dpt import DepthAnythingV2
import utils.orientation as orient
import utils.coordinates as coord
from utils.camera import img_from_device, denormalize, view_frame_from_device_frame
import math

print(f"Cuda available: {torch.cuda.is_available()}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

focal_length_y = 910
focal_length_x = 910


# 7, 24, 19
# 19 -> has a curve
# 9 -> has a huge curve and a lot of cars
# 24 -> has a curve and a lot of cars
# 27 -> very close objects

SEGMENT_TO_PROCESS = 19
MAX_TOTAL_FRAMES_TO_PROCESS = 300   # Set your total frame limit here, 0 to use segment counter

SEGMENT_TO_PROCESS__PATH = Path("/home/adrian/foxglove_work/6_comma2k19/DATA/Chunk_5/99c94dc769b5d96e|2018-07-03--14-38-26/" + str(SEGMENT_TO_PROCESS))




MAX_TOTAL_SEGMENTS_TO_PROCESS = 300  

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # or 'vits', 'vitb'
dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 250 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


# Path to the dataset data
path_to_data = Path("/home/adrian/foxglove_work/6_comma2k19/DATA")
output_mcap_path = "/home/adrian/foxglove_work/6_comma2k19/"


output_mcap_file = output_mcap_path + "comma2k19_dataset_SEG" + str(SEGMENT_TO_PROCESS) + "_F0_F" + str(MAX_TOTAL_FRAMES_TO_PROCESS) + ".mcap"



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

def invert_quaternion(quat):
    """Inverte un quaternion Hamiltoniano."""
    return np.array([quat[0], quat[1], quat[2], -quat[3]])
  
def quaternion_multiply(q1, q2):
    """Multiplica dos quaternions q1 y q2 en formato Hamiltoniano [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w])
  
def process_log_data(name, values, time, writer, channel_id_dict, max_timestamp_ns):
  print(f"[{name}] value_shape: {values.shape}, time_size: {time.size}, vs: {values.size}")
  print(f"Max timestamp_s = {max_timestamp_ns /  1_000_000_000}")

  i=0
  for timestamp_s, value in zip(time, values):
    timestamp_ns = seconds_to_nanoseconds(timestamp_s)
    timestamp_iso8601 = timestamp_ns_to_iso8601(timestamp_ns)
    timestep_sec = timestamp_ns // 1_000_000_000
    timestep_nsec = timestamp_ns % 1_000_000_000
    
    
    if name == "speed":
      #print(f"value = {value}, timestamp = {timestamp_ns}")
      message = {
              "timestamp": timestamp_iso8601,
              "value": float(value)
      }
      writer.add_message(
            channel_id=channel_id_dict[name],
            log_time=timestamp_ns,
            data=json.dumps(message).encode('utf-8'),
            publish_time=timestamp_ns,
        )
    if name == "wheel_speed":
      #print(f"value = {value}, timestamp = {timestamp_ns}")
      message = {
              "timestamp": timestamp_iso8601,
              "FR": float(value[0]),
              "FL": float(value[1]),
              "RR": float(value[2]),
              "RL": float(value[3])
      }
      writer.add_message(
            channel_id=channel_id_dict[name],
            log_time=timestamp_ns,
            data=json.dumps(message).encode('utf-8'),
            publish_time=timestamp_ns,
        )
      
    
    if name == "live_gnss_qcom":
            
      message = {
        "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
        "frame" : "map",
        "latitude" : value[0],
        "longitude" : value[1],  
        "altitude" : value[4]
      }    
      writer.add_message(
            channel_id=channel_id_dict[name],
            log_time=timestamp_ns,
            data=json.dumps(message).encode('utf-8'),
            publish_time=timestamp_ns,
        )
  
    
    if name == "live_gnss_ublox":
      message = {
        "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
        "frame" : "map",
        "latitude" : value[0],
        "longitude" : value[1],  
        "altitude" : value[4]
      }    
      writer.add_message(
            channel_id=channel_id_dict[name],
            log_time=timestamp_ns,
            data=json.dumps(message).encode('utf-8'),
            publish_time=timestamp_ns,
        )
     
    i +=1
    if timestamp_ns > max_timestamp_ns:
      if name in channel_id_dict:
        print(f"[{name}] N_processed: {i}")
      break

def process_segment(segment, total_frames_limit, total_frames_processed, writer, channel_id_dict):
    # We search for the video
    video_file = None
    global_pos_dir = None
    processed_log_dir = None
    frame_times = None
    frame_positions = None
    frame_orientations = None
    max_timestamp_ns = 0
    
    position_ref = None
    orientation_ref = None
    orientation_ref_inv = None
    for file in segment.iterdir():   
        if file.suffix == ".hevc":  # We check if the file is a .hevc video
            video_file = file
        if file.name == "global_pose":
          global_pos_dir = file 
        if file.name == "processed_log":
          processed_log_dir = file
          
    for subfile in global_pos_dir.iterdir():
      if subfile.name == "frame_times":
        frame_times = np.load(str(subfile))
        max_timestamp_ns = seconds_to_nanoseconds(frame_times[total_frames_limit])
        print(f"Max timestamp_ns: {max_timestamp_ns}")
      if subfile.name == "frame_positions":
        frame_positions = np.load(str(subfile))
        position_ref = np.array(frame_positions[0])
        print(F"Position Ref = {position_ref}")
        
      if subfile.name == "frame_orientations":
        frame_orientations = np.load(str(subfile))
        orientation_ref = np.array(frame_orientations[0])
        print(F"Orientation Ref = {orientation_ref}")
    for subfile in processed_log_dir.iterdir():
      if subfile.name == "CAN" or subfile.name == "GNSS":
        for can_subfile in subfile.iterdir(): 
          if can_subfile.name == "raw_can":
            break
          times = np.load(str(can_subfile) + "/" + "t")
          values = np.load(str(can_subfile) + "/" + "value")
          name = can_subfile.name
          process_log_data(name, values, times, writer, channel_id_dict,max_timestamp_ns)
               
    
    orientation_ref_inv = invert_quaternion(orientation_ref)
    
                
    
    cap = cv2.VideoCapture(str(video_file))
    
    # Initialize frame counter for the current segment
    frame_number = 0
    initial_timestamp_s = frame_times[frame_number]
    while cap.isOpened():
        print(f"frame number: {frame_number}")
        ret, frame = cap.read()
  
        # Break the loop if no frame is returned (end of video)
        if not ret:
            print("Reached the end of the video.")
            break
        
        # Process the frame timestamp
        frame_timestamp_seconds = frame_times[frame_number]
        
        frame_timestamp_nanoseconds = seconds_to_nanoseconds(frame_timestamp_seconds)
        frame_timestamp_iso = timestamp_ns_to_iso8601(frame_timestamp_nanoseconds)
        timestep_sec = frame_timestamp_nanoseconds // 1_000_000_000
        timestep_nsec = frame_timestamp_nanoseconds % 1_000_000_000
        height, width, _ = frame.shape
      
        #prediction
        depth = model.infer_image(frame, height) # HxW raw depth map in numpy
        depth_image_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_uint8 = np.uint8(depth_image_normalized)

        # Aplicamos un mapa de colores, como 'COLORMAP_JET'
        depth_image_colored = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
        
        # Resize depth prediction to match the original image size
        resized_depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Image array to png
        _, buffer = cv2.imencode('.png', frame)
        _, buffer_depth = cv2.imencode('.png', resized_depth)
        _, buffer_depth_colored = cv2.imencode('.png', depth_image_colored)
        
        
        
        
        ## we crop the images
        frame_croped = frame[300:874,0:1164]
        resized_depth_croped = resized_depth[300:874,0:1164]
        height, width, _ = frame_croped.shape
        
        # to base 64
        img_str = base64.b64encode(buffer).decode('utf-8')
        depth_str = base64.b64encode(buffer_depth).decode('utf-8')
        depth_colored_str = base64.b64encode(buffer_depth_colored).decode('utf-8')
        
        # Creation of the pointcloud
        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = ((x - width / 2) / focal_length_x)
        y = ((y - height / 2) / focal_length_y)
        z = - (np.array(resized_depth_croped))
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3) 
        
        x = - points[:, 2]
        y = - points[:, 0]
        z = points[:, 1]
        
        x_threshold = 30  # Cambia este valor según tus necesidades
        # Filtra los puntos para mantener solo los que cumplen la condición x >= x_threshold
        mask = x >= x_threshold
        x = - points[:, 2][mask]
        y = - points[:, 0][mask]
        z = points[:, 1][mask]

        colors = np.array(frame_croped).reshape(-1, 3) / 255.0
        b_values = (colors[:,0])[mask]
        g_values = (colors[:,1])[mask]
        r_values = (colors[:,2])[mask]
        ca_values = np.ones_like(b_values)
        
        # Stack the points
        pcl = np.vstack((x, y, z,r_values,g_values,b_values,ca_values)).T
        points_flat = pcl.astype(np.float32).tobytes()
        data_encoded = base64.b64encode(points_flat).decode('utf-8')
        fields = [
                {"name": "x", "offset": 0, "type": 7},
                {"name": "y", "offset": 4, "type": 7},
                {"name": "z", "offset": 8, "type": 7},
                {"name": "red", "offset": 12, "type": 7},
                {"name": "green", "offset": 16, "type": 7},
                {"name": "blue", "offset": 20, "type": 7},
                {"name": "alpha", "offset": 24, "type": 7},
            ]
                        
        pointcloud_message = {
          "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
          "frame_id": "lidar",  # You can replace this with your actual frame of reference  
          "point_stride": 28,
          "fields": fields,
          "data": data_encoded  
        }
        
        # Schema for foxglove.CompressedImage
        image_message = {
                    "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
                    "frame_id": "frame",
                    "format": "png",
                    "data": img_str
                }
        
        depth_message = {
                    "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
                    "frame_id": "frame",
                    "format": "png",
                    "data": depth_str
                }
        
        depth_colored_message = {
                    "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
                    "frame_id": "frame",
                    "format": "png",
                    "data": depth_colored_str
                }
    
        # We add the message
        writer.add_message(
            channel_id=channel_id_dict["image"],
            log_time=frame_timestamp_nanoseconds,
            data=json.dumps(image_message).encode('utf-8'),
            publish_time=frame_timestamp_nanoseconds,
        )
        
        writer.add_message(
            channel_id=channel_id_dict["depth"],
            log_time=frame_timestamp_nanoseconds,
            data=json.dumps(depth_message).encode('utf-8'),
            publish_time=frame_timestamp_nanoseconds,
        )
        
        writer.add_message(
            channel_id=channel_id_dict["depth_colored"],
            log_time=frame_timestamp_nanoseconds,
            data=json.dumps(depth_colored_message).encode('utf-8'),
            publish_time=frame_timestamp_nanoseconds,
        )
        
        writer.add_message(
            channel_id=channel_id_dict["pcl"],
            log_time=frame_timestamp_nanoseconds,
            data=json.dumps(pointcloud_message).encode('utf-8'),
            publish_time=frame_timestamp_nanoseconds,
        )
        
        # We publish the transforms
        
        
        
        euler_angles_ned_deg = (180/3.1415)*orient.ned_euler_from_ecef(frame_positions[frame_number], orient.euler_from_quat(frame_orientations))
        ecef_from_local = orient.rot_from_quat(frame_orientations[frame_number])
        local_from_ecef = ecef_from_local.T
        frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[frame_number])
        
       
        end = frame_number + 100
        ini = frame_number + 1
        if (end > frame_positions_local.size):
          end = frame_positions_local.size
        
        device_path = frame_positions_local[ini:end]
        device_path_l = device_path + np.array([0, 0, 1.2])                                                                    
        device_path_r = device_path + np.array([0, 0, 1.2])                                                                    
        device_path_l[:,1] -= 1                                                                                               
        device_path_r[:,1] += 1
        img_points_norm_l = img_from_device(device_path_l)
        img_points_norm_r = img_from_device(device_path_r)
        img_pts_l = denormalize(img_points_norm_l)
        img_pts_r = denormalize(img_points_norm_r)
        
        
        # filter out things rejected along the way
        valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
        img_pts_l = img_pts_l[valid].astype(int)
        img_pts_r = img_pts_r[valid].astype(int)
        points = []
        for i in range(1, len(img_pts_l)):
          u1,v1,u2,v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
          u3,v3,u4,v4 = np.append(img_pts_l[i], img_pts_r[i])
          
          point_annotation = {
                      "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
                      "type" : 2,
                      "points" : [{"x": np.float64(u1), "y": np.float64(v1)},
                                  {"x": np.float64(u2), "y": np.float64(v2)},
                                  {"x": np.float64(u4), "y": np.float64(v4)},
                                  {"x": np.float64(u3), "y": np.float64(v3)}],
                      #"outline_color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.5},
                      "outline_colors":[{"r": 0.0, "g": 1.0, "b": 0.0, "a": 0.8},
                                        {"r": 0.0, "g": 1.0, "b": 0.0, "a": 0.8},
                                        {"r": 0.0, "g": 1.0, "b": 0.0, "a": 0.8},
                                        {"r": 0.0, "g": 1.0, "b": 0.0, "a": 0.8}],
                      "fill_color": {"r": 128/255, "g": 0.0, "b": 1.0, "a": 0.4},
                      "thickness" : 1,
                    }
          
          points.append(point_annotation)
          
        annotation_message = {
              "points" : points
          }
        
    
        writer.add_message(
            channel_id=channel_id_dict["annotations"],
            log_time=frame_timestamp_nanoseconds,
            data=json.dumps(annotation_message).encode('utf-8'),
            publish_time=frame_timestamp_nanoseconds,
        )

  
        tf_world_car = {
              "timestamp": {
                  "sec": timestep_sec,
                  "nsec": timestep_nsec
              },
              "parent_frame_id": "world",  # Cambia "world" si tu sistema de referencia es diferente
              "child_frame_id": "car",
              "translation": {
                  "x": 0,
                  "y": 0,
                  "z": 0
              },
              "rotation": {
                  "x": 0,
                  "y": 0,
                  "z": 0,
                  "w": 0
              }
          }
     
        tf_car_lidar = {
                  "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
                  "parent_frame_id": "car",  # You can replace this with your actual frame of reference
                  "child_frame_id": "lidar",
                  "translation": {"x" : -15, "y" : 0, "z" : 10},
                  "rotation": {"x" :0 , "y" : 0.1305262, "z" : 0, "w" : 0.991444}  
                }
        
        writer.add_message(
                        channel_id=channel_id_dict["tf_world_car"],
                        log_time=frame_timestamp_nanoseconds,
                        data=json.dumps(tf_world_car).encode('utf-8'),
                        publish_time=frame_timestamp_nanoseconds,
                    )
        
        writer.add_message(
                        channel_id=channel_id_dict["tf_car_lidar"],
                        log_time=frame_timestamp_nanoseconds,
                        data=json.dumps(tf_car_lidar).encode('utf-8'),
                        publish_time=frame_timestamp_nanoseconds,
                    )
        
        # Increment frame counter
        frame_number += 1

        # Update total frame count
        total_frames_processed[0] += 1
        
        # Check if the total limit has been reached
        if total_frames_processed[0] >= total_frames_limit:
            print(f"Reached the limit of {total_frames_limit} total frames processed.")
            break
        #elapsed_time_s = frame_timestamp_seconds - initial_timestamp_s
        #print(f"Elapsed_time_frames: {elapsed_time_s}")
        #if elapsed_time_s > max_timestamp:
        #  break



writer = Writer(output_mcap_file)
writer.start()
channel_id_dict = {}

compressed_image_schema = open_json_schema("json_schemas/CompressedImage.json") # Schema for the topic
compressed_image_schema_id = writer.register_schema(                            # Schema id
        name="foxglove.CompressedImage",
        encoding="jsonschema",
        data=json.dumps(compressed_image_schema).encode(),
    )

image_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=compressed_image_schema_id,
    topic="camera/rgb_camera",
    message_encoding="json",
)

channel_id_dict["image"] = image_channel_id

depth_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=compressed_image_schema_id,
    topic="camera/depth_camera",
    message_encoding="json",
)

channel_id_dict["depth"] = depth_channel_id
depth_colored_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=compressed_image_schema_id,
    topic="camera/depth_camera_colored",
    message_encoding="json",
)
channel_id_dict["depth_colored"] = depth_colored_channel_id



pcl_schema = open_json_schema("json_schemas/PointCloud.json") # Schema for the topic
pcl_schema_id = writer.register_schema(                            # Schema id
        name="foxglove.PointCloud",
        encoding="jsonschema",
        data=json.dumps(pcl_schema).encode(),
    )
pcl_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=pcl_schema_id,
    topic="pointcloud",
    message_encoding="json",
)
channel_id_dict["pcl"] = pcl_channel_id

# For floats
float_schema = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "value": {"type": "number"}
    },
    "required": ["timestamp", "value"]
}

float_schema_id = writer.register_schema(
            name="float",
            encoding="jsonschema",
            data=json.dumps(float_schema).encode(),
        )

speed_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=float_schema_id,
    topic="CAN/car_speed",
    message_encoding="json",
)



channel_id_dict["speed"] = speed_channel_id

# For wheel_speed
float_schema = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "FR": {"type": "number"},
        "FL": {"type": "number"},
        "RR": {"type": "number"},
        "RL": {"type": "number"}
        
    },
    "required": ["timestamp"]
}

wheel_speed_schema_id = writer.register_schema(
            name="wheel_speed",
            encoding="jsonschema",
            data=json.dumps(float_schema).encode(),
        )

wheel_speed_channel_id = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=wheel_speed_schema_id,
    topic="CAN/wheel_speed",
    message_encoding="json",
)

channel_id_dict["wheel_speed"] = wheel_speed_channel_id
# For floats

# Frame transforms
frame_schema = open_json_schema("json_schemas/FrameTransform.json")
frame_schema_id = writer.register_schema(
        name="foxglove.FrameTransform",
        encoding="jsonschema",
        data=json.dumps(frame_schema).encode(),
    )

frame_world_car= writer.register_channel(
    schema_id=frame_schema_id,
    topic="tf/tf_world_car",
    message_encoding="json",
)

frame_car_lidar = writer.register_channel(
    schema_id=frame_schema_id,
    topic="tf/tf_car_lidar",
    message_encoding="json",
)
channel_id_dict["tf_world_car"] = frame_world_car
channel_id_dict["tf_car_lidar"] = frame_car_lidar

gps_schema = open_json_schema("json_schemas/LocationFix.json")
gps_schema_id = writer.register_schema(
  name="foxglove.LocationFix",
  encoding="jsonschema",
  data=json.dumps(gps_schema).encode(),
)

live_gnss_qcom_channel_id = writer.register_channel(
    schema_id=gps_schema_id,
    topic="GNSS/live_gnss_qcom",
    message_encoding="json",
)
channel_id_dict["live_gnss_qcom"] = live_gnss_qcom_channel_id
live_gnss_ublox_channel_id = writer.register_channel(
    schema_id=gps_schema_id,
    topic="GNSS/live_gnss_ublox",
    message_encoding="json",
)
channel_id_dict["live_gnss_ublox"] = live_gnss_ublox_channel_id

annotations_schema= open_json_schema("json_schemas/ImageAnnotations.json")

annotations_schema_id = writer.register_schema(
  name="foxglove.ImageAnnotations",
  encoding="jsonschema",
  data=json.dumps(annotations_schema).encode(),
)
annotations_channel_id = writer.register_channel(
    schema_id=annotations_schema_id,
    topic="path_image_annotations",
    message_encoding="json",
)
channel_id_dict["annotations"] = annotations_channel_id


with open(output_mcap_file, 'wb') as mcap_file:

  #for chunk in path_to_data.iterdir(): 
  #  for route in chunk.iterdir():
  #    for segment in route.iterdir():
  #      segments.append(segment)
  #      
  #  # Sort segments based on their numeric names
  #  
  total_frames_processed = [0]  # Initialize total frames processed as a list
  #  
  #  segments.sort(key=lambda x: int(x.name))
  #  print(segments[SEGMENT_TO_PROCESS])
  process_segment(SEGMENT_TO_PROCESS__PATH, MAX_TOTAL_FRAMES_TO_PROCESS, total_frames_processed, writer, channel_id_dict)
    
    
    #        segments.append(segment)
    #for route in chunk.iterdir():
    #    print(f"Processing route: {route.name} in chunk: {chunk.name}")
    #    segments = []
    #    
    #    for segment in route.iterdir():
    #        segments.append(segment)
    #    
    #    # Sort segments based on their numeric names
    #    segments.sort(key=lambda x: int(x.name))
    #    
    #    total_frames_processed = [0]  # Initialize total frames processed as a list
    #    total_segments_processed = 0
    #    
    #    for segment in segments:
    #        print(f"Segment: {segment.name}")
    #        process_segment(segment, MAX_TOTAL_FRAMES_TO_PROCESS, total_frames_processed, writer, channel_id_dict)
    #        
    #        # Break if the total frame limit has been reached
    #        if MAX_TOTAL_FRAMES_TO_PROCESS != 0:
    #          if total_frames_processed[0] >= MAX_TOTAL_FRAMES_TO_PROCESS:
    #              print(f"Reached the limit of {MAX_TOTAL_FRAMES_TO_PROCESS} total frames processed.")
    #              break
    #        else:
    #          total_segments_processed+=1
    #          print(f"Number of processed segments = {total_segments_processed}")
    #    if MAX_TOTAL_FRAMES_TO_PROCESS != 0:
    #      if total_frames_processed[0] >= MAX_TOTAL_FRAMES_TO_PROCESS:
    #          break
        
writer.finish()