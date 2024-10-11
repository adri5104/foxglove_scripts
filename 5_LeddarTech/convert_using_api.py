from pathlib import Path
import json
from mcap.writer import Writer
from datetime import datetime, timezone
import cv2
from scipy.spatial.transform import Rotation as R
import base64
import numpy as np
import pandas as pd
from pioneer.das.api.platform import Platform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dictionary that assigns color to a label. Used in several datasets
colors_dict = {
    # flat
    "road": {"r": 128/255, "g": 64/255, "b": 128/255, "a": 0.0},
    "sidewalk": {"r": 244/255, "g": 35/255, "b": 232/255, "a": 0.0},
    "parking": {"r": 250/255, "g": 170/255, "b": 160/255, "a": 0.0},
    "rail track": {"r": 230/255, "g": 150/255, "b": 140/255, "a": 0.0},

    # human (green with alpha = 1)
    "person": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0},
    "pedestrian": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0},
    "rider": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0},

    # vehicle (red with alpha = 1)
    "car": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "truck": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "bus": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "on rails": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "motorcycle": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "bicycle": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "caravan": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "trailer": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "ego vehicle": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "license plate": {"r": 1.0, "g": 1.0, "b": 0.0, "a": 1.0},  # Yellow with alpha = 1

    # construction (purple with alpha = 1)
    "building": {"r": 128/255, "g": 0.0, "b": 128/255, "a": 1.0},
    "wall": {"r": 128/255, "g": 0.0, "b": 128/255, "a": 1.0},
    "fence": {"r": 128/255, "g": 0.0, "b": 128/255, "a": 1.0},
    "guard rail": {"r": 128/255, "g": 0.0, "b": 128/255, "a": 1.0},
    "bridge": {"r": 128/255, "g": 0.0, "b": 128/255, "a": 1.0},
    "tunnel": {"r": 128/255, "g": 0.0, "b": 128/255, "a": 1.0},

    # object (blue with alpha = 1)
    "pole": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "pole group": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "traffic sign": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "traffic light": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "traffic cone": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},

    # nature (alpha = 0)
    "vegetation": {"r": 107/255, "g": 142/255, "b": 35/255, "a": 0.0},
    "terrain": {"r": 152/255, "g": 251/255, "b": 152/255, "a": 0.0},

    # sky (alpha = 0)
    "sky": {"r": 70/255, "g": 130/255, "b": 180/255, "a": 0.0},

    # void (alpha = 0)
    "ground": {"r": 0/255, "g": 0/255, "b": 0/255, "a": 0.0},
    "dynamic": {"r": 111/255, "g": 74/255, "b": 0/255, "a": 0.0},
    "static": {"r": 81/255, "g": 0/255, "b": 81/255, "a": 0.0}
}

output_mcap_file = "/home/adrian/foxglove_work/5_LeddarTech_dataset/output.mcap"

# Function to open json schemas
def open_json_schema(file_name):
  path = Path(__file__).parent / file_name
  with open(path, "r") as schema_f:
    schema = json.load(schema_f)
    return schema
  
def timestamp_ns_to_iso8601(timestamp_ns):
    return datetime.fromtimestamp(int(timestamp_ns) / 1e9, tz=timezone.utc).isoformat()
  
  


writer = Writer(output_mcap_file)
writer.start()

# ================= Topic definitions ================
# Image data
compressed_image_schema = open_json_schema("json_schemas/CompressedImage.json") # Schema for the topic
compressed_image_schema_id = writer.register_schema(                            # Schema id
        name="foxglove.CompressedImage",
        encoding="jsonschema",
        data=json.dumps(compressed_image_schema).encode(),
    )

image_channel_id_c = writer.register_channel(                                   # Channel using the schema id (Ros topic)
    schema_id=compressed_image_schema_id,
    topic="rgb_camera_center",
    message_encoding="json",
)

image_channel_id_r = writer.register_channel(
  schema_id=compressed_image_schema_id,
  topic="rgb_camera_right",
  message_encoding="json",
)
image_channel_id_l = writer.register_channel(
    schema_id=compressed_image_schema_id,
    topic="rgb_camera_left",
    message_encoding="json",
)

image_channel_id_cyl = writer.register_channel(
    schema_id=compressed_image_schema_id,
    topic="rgb_camera_cyl",
    message_encoding="json",
)

# Camera calibration data
camera_calibration_schema = open_json_schema("json_schemas/CameraCalibration.json")
camera_calibration_schema_id = writer.register_schema(
        name="foxglove.CameraCalibration",
        encoding="jsonschema",
        data=json.dumps(camera_calibration_schema).encode(),
    )

camera_calibration_id_c = writer.register_channel(
    schema_id=camera_calibration_schema_id,
    topic="cameracalibration_center",
    message_encoding="json",
)

camera_calibration_id_r = writer.register_channel(
    schema_id=camera_calibration_schema_id,
    topic="cameracalibration_right",
    message_encoding="json",
)

camera_calibration_id_l = writer.register_channel(
    schema_id=camera_calibration_schema_id,
    topic="cameracalibration_left",
    message_encoding="json",
)

# Frame transforms
frame_schema = open_json_schema("json_schemas/FrameTransform.json")
frame_schema_id = writer.register_schema(
        name="foxglove.FrameTransform",
        encoding="jsonschema",
        data=json.dumps(frame_schema).encode(),
    )

frame_idd = writer.register_channel(
    schema_id=frame_schema_id,
    topic="tf",
    message_encoding="json",
)

frame_idd_car = writer.register_channel(
    schema_id=frame_schema_id,
    topic="tf_car",
    message_encoding="json",
)

frame_idd_radar = writer.register_channel(
    schema_id=frame_schema_id,
    topic="tf_radar",
    message_encoding="json",
)

# Image annotations
annotation_schema = open_json_schema("json_schemas/ImageAnnotations.json")
annotation_schema_id = writer.register_schema(
        name="foxglove.ImageAnnotations",
        encoding="jsonschema",
        data=json.dumps(annotation_schema).encode(),
    )

annotation_id = writer.register_channel(
    schema_id=annotation_schema_id,
    topic="cyl_img_annotations",
    message_encoding="json",
)

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

speed_channel_id = writer.register_channel(
    schema_id=0,
    topic="CANdata",
    message_encoding="json",
)

# Pointcloud data
pcl_schema = open_json_schema("json_schemas/PointCloud.json")
pcl_schema_id = writer.register_schema(
    name="foxglove.PointCloud",
    encoding="jsonschema",
    data=json.dumps(pcl_schema).encode(),
)
pcl_channel_id = writer.register_channel(
    schema_id=pcl_schema_id,
    topic="pointcloud_pixell",
    message_encoding="json",
)

pcl_channel_id2 = writer.register_channel(
    schema_id=pcl_schema_id,
    topic="pointcloud_ouster",
    message_encoding="json",
)

pcl_channel_id3 = writer.register_channel(
    schema_id=pcl_schema_id,
    topic="pointcloud_radar",
    message_encoding="json",
)

# Entity data (3D boxes)
entity_schema  = open_json_schema("json_schemas/SceneUpdate.json")

entity_schema_id  = writer.register_schema(
    name="foxglove.SceneUpdate",
    encoding="jsonschema",
    data=json.dumps(entity_schema).encode(),
)
entity_channel_id = writer.register_channel(
    schema_id=entity_schema_id,
    topic="cubes",
    message_encoding="json",
)

# GPS data
gps_schema  = open_json_schema("json_schemas/LocationFix.json")

gps_schema_id  = writer.register_schema(
    name="foxglove.LocationFix",
    encoding="jsonschema",
    data=json.dumps(gps_schema).encode(),
)
gps_channel_id = writer.register_channel(
    schema_id=gps_schema_id,
    topic="location",
    message_encoding="json",
)

# ===================== Processing of the data =========================================

pf = Platform('/home/adrian/foxglove_work/5_LeddarTech_dataset/DATA')
sync = pf.synchronized()
print(pf.extrinsics)

#we get the sensors
camera_center = pf.sensors['flir_bfc']
camera_right = pf.sensors['flir_bfr']
camera_left = pf.sensors['flir_bfl']
pixell_lidar = pf.sensors['pixell_bfc']
sbgekinox_bcc = pf.sensors['sbgekinox_bcc']
camera_center_cyl = pf.sensors['flir_bbfc']

# initial frame to process
ini = 0     

# final frame to process
fin = 200

#output file name
output_mcap_file = "/home/adrian/foxglove_work/5_LeddarTech_dataset/output_" + str(ini) + "_" + str(fin) + ".mcap"
  
# We start to process the data
with open(output_mcap_file, 'wb') as mcap_file:
  entities = []
  timestamp_init = 0
  
  for index in range(ini,fin):
    
    # ================== CAMERA cyl ==================
    image = camera_center_cyl['img'][index].undistort_image()
    timestamp = camera_center_cyl['poly2d-detectron-undistort'][index].raw["timestamp"]*1000
    iso_timestamp = timestamp_ns_to_iso8601(timestamp)
    timestamp_ns = int(timestamp)
    
    
    # Image array to png
    _, buffer = cv2.imencode('.png', image)

    # to base 64
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Schema for foxglove.CompressedImage
    image_message = {
                "timestamp": iso_timestamp,
                "frame_id": "camera_frame",
                "format": "png",
                "data": img_str
            }
    
    # We add the message
    writer.add_message(
        channel_id=image_channel_id_cyl,
        log_time=timestamp_ns,
        data=json.dumps(image_message).encode('utf-8'),
        publish_time=timestamp_ns,
    )
    
    
     
    
    # ================== CAMERA CENTER ==================
    timestamp = image_sample = sync[index]['flir_bfc_img'].timestamp*1000
    timestamp_init = timestamp
    image_raw = image_sample = sync[index]['flir_bfc_img'].raw
    iso_timestamp = timestamp_ns_to_iso8601(timestamp)
    timestamp_ns = int(timestamp)
    timestep_sec = timestamp_ns // 1_000_000_000
    timestep_nsec = timestamp_ns % 1_000_000_000
    image = sync[index]['flir_bfc_img']
    _, buffer = cv2.imencode('.png', image_raw)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_message = {
                "timestamp": iso_timestamp,
                "frame_id": "camera_frame",
                "format": "png",
                "data": img_str
            }

    writer.add_message(
        channel_id=image_channel_id_c,
        log_time=timestamp_ns,
        data=json.dumps(image_message).encode('utf-8'),
        publish_time=timestamp_ns,
    )
    
    calibration_message = {
      "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
      "frame_id" : "camera_frame",
      "width" : 1440,
      "height" : 1080,
      "distorsion_model" : "plumb_bob",
      "D" : image.distortion_coeffs.flatten().tolist(),
      "K" : image.und_camera_matrix.flatten().tolist(),
      "P" : image.extrinsics['flir_bfc'][[0,1,2],:].flatten().tolist()
    }
    writer.add_message(
        channel_id=camera_calibration_id_c,
        log_time=timestamp_ns,
        data=json.dumps(calibration_message).encode('utf-8'),
        publish_time=timestamp_ns,
    )
 
    # ================== CAMERA RIGHT ==================
    timestamp  = sync[index]['flir_bfr_img'].timestamp*1000
    image_raw  = sync[index]['flir_bfr_img'].raw
    iso_timestamp = timestamp_ns_to_iso8601(timestamp)
    timestamp_ns = int(timestamp)
    image = sync[index]['flir_bfr_img']
    
    _, buffer = cv2.imencode('.png', image_raw)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_message = {
                "timestamp": iso_timestamp,
                "frame_id": "camera_frame",
                "format": "png",
                "data": img_str
            }

    writer.add_message(
        channel_id=image_channel_id_r,
        log_time=timestamp_ns,
        data=json.dumps(image_message).encode('utf-8'),
        publish_time=timestamp_ns,
    )  
    
    calibration_message = {
      "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
      "frame_id" : "camera_frame",
      "width" : 1920,
      "height" : 1080,
      "distorsion_model" : "plumb_bob",
      "D" : image.distortion_coeffs.flatten().tolist(),
      "K" : image.intrinsics['matrix'].flatten().tolist(),
      "P" : image.extrinsics['flir_bfr'][[0,1,2],:].flatten().tolist()
    }
    
    
    # ================== CAMERA LEFT ==================
    timestamp = sync[index]['flir_bfl_img'].timestamp*1000
    image_raw = sync[index]['flir_bfl_img'].raw
    iso_timestamp = timestamp_ns_to_iso8601(timestamp)
    timestamp_ns = int(timestamp)
    image = sync[index]['flir_bfl_img']
    _, buffer = cv2.imencode('.png', image_raw)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_message = {
                "timestamp": iso_timestamp,
                "frame_id": "camera_frame",
                "format": "png",
                "data": img_str
            }
   
    writer.add_message(
        channel_id=image_channel_id_l,
        log_time=timestamp_ns,
        data=json.dumps(image_message).encode('utf-8'),
        publish_time=timestamp_ns,
    )
    
    calibration_message = {
      "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
      "frame_id" : "camera_frame",
      "width" : 1920,
      "height" : 1080,
      "distorsion_model" : "plumb_bob",
      "D" : image.distortion_coeffs.flatten().tolist(),
      "K" : image.intrinsics['matrix'].flatten().tolist(),
      "P" : image.extrinsics['flir_bfl'][[0,1,2],:].flatten().tolist()
    }
    
    writer.add_message(
        channel_id=camera_calibration_id_l,
        log_time=timestamp_ns,
        data=json.dumps(calibration_message).encode('utf-8'),
        publish_time=timestamp_ns,
    )
    
    # ================== SPEED ==================
    gasPedal = sync[index]['peakcan_fcc_GasPedal'].raw[1] 
    wheelSpeed = sync[index]['peakcan_fcc_WheelSpeeds'].raw.item()
    brakeModule = sync[index]['peakcan_fcc_BrakeModule'].raw.item()
    steering = sync[index]['peakcan_fcc_SteerAngleSensor'].raw.item()
    speed = sync[index]['sbgekinox_bcc_navposvel'].raw.item()
    speed_message = {
                      "timestamp": timestamp_ns,
                      "WheelSpeed": {"FR": wheelSpeed[0] ,"FL": wheelSpeed[1],"RR": wheelSpeed[2],"RL": wheelSpeed[3]},
                      "GasPedal" : gasPedal,
                      "Brake" : {"Pressure": brakeModule[0], "BrakePressed" : brakeModule[1]},
                      "SteeringAngle" : steering[0],
                      "CarSpeed" : speed[1]
                  }
    writer.add_message(
                      channel_id=speed_channel_id,
                      log_time=timestamp_ns,
                      data=json.dumps(speed_message).encode('utf-8'),
                      publish_time=sync[index]['sbgekinox_bcc_gps1vel'].timestamp,
                  )
  
    # ================== POINTCLOUD (Pixel center)==================
    echoes = sync[index]['pixell_bfc_ech']
    
    x=echoes.point_cloud()[:,0]
    y=echoes.point_cloud()[:,1]
    z=echoes.point_cloud()[:,2]
    
    # Stack the points
    points = np.vstack((x, y, z)).T
    points_flat = points.astype(np.float32).tobytes()
    data_encoded = base64.b64encode(points_flat).decode('utf-8')
    fields = [
        {"name": "x", "offset": 0, "type": 7},
        {"name": "y", "offset": 4, "type": 7},
        {"name": "z", "offset": 8, "type": 7},
      ]
    
    pointcloud_message = {
              "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
              "frame_id": "lidar",  # You can replace this with your actual frame of reference
              
              "point_stride":12,
              "fields": fields,
              "data": data_encoded  
            }
    writer.add_message(
                    channel_id=pcl_channel_id,
                    log_time=timestamp_ns,
                    data=json.dumps(pointcloud_message).encode('utf-8'),
                    publish_time=timestamp_ns,
                )  
    
    # ================== POINTCLOUD (Ouster)==================
    echoes = sync[index]['ouster64_bfc_xyzit']
    
    x=echoes.point_cloud()[:,0]
    y=echoes.point_cloud()[:,1]
    z=echoes.point_cloud()[:,2]
    
    # Stack the points
    points = np.vstack((x, y, z)).T
    points_flat = points.astype(np.float32).tobytes()
    data_encoded = base64.b64encode(points_flat).decode('utf-8')
    fields = [
        {"name": "x", "offset": 0, "type": 7},
        {"name": "y", "offset": 4, "type": 7},
        {"name": "z", "offset": 8, "type": 7},
      ]
    
    pointcloud_message = {
              "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
              "frame_id": "lidar",  # You can replace this with your actual frame of reference
              
              "point_stride":12,
              "fields": fields,
              "data": data_encoded  
            }
    writer.add_message(
                    channel_id=pcl_channel_id2,
                    log_time=timestamp_ns,
                    data=json.dumps(pointcloud_message).encode('utf-8'),
                    publish_time=timestamp_ns,
                )  
    
    # ================== POINTCLOUD (Radar)==================
    echoes = sync[index]['radarTI_bfc_xyzvcfar']
    
    x=echoes.point_cloud()[:,0]
    y=echoes.point_cloud()[:,1]
    z=echoes.point_cloud()[:,2]
    # Stack the points
    points = np.vstack((x, y, z)).T
    points_flat = points.astype(np.float32).tobytes()
    data_encoded = base64.b64encode(points_flat).decode('utf-8')
    fields = [
        {"name": "x", "offset": 0, "type": 7},
        {"name": "y", "offset": 4, "type": 7},
        {"name": "z", "offset": 8, "type": 7},
      ]
    pointcloud_message = {
              "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
              "frame_id": "radar",  # You can replace this with your actual frame of reference
              
              "point_stride":12,
              "fields": fields,
              "data": data_encoded  
            }
    writer.add_message(
                    channel_id=pcl_channel_id3,
                    log_time=timestamp_ns,
                    data=json.dumps(pointcloud_message).encode('utf-8'),
                    publish_time=timestamp_ns,
                )  
    
    
    # ================== Transforms ==================
    tf_flir_lidar = {
              "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
              "parent_frame_id": "car",  # You can replace this with your actual frame of reference
              "child_frame_id": "lidar",
              "translation": {"x" : 0, "y" : 0, "z" : 0.3},
              "rotation": {"x" : 0, "y" : 0, "z" : 0, "w" : 0}  
            }
    
    tf_flir_car = {
              "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
              "parent_frame_id": "world",  # You can replace this with your actual frame of reference
              "child_frame_id": "car",
              "translation": {"x" : 3.6, "y" : 0, "z" : 0},
              "rotation": {"x" : 0, "y" : 0, "z" : 0, "w" : 0}  
            }
    
    tf_lidar_radar = {
              "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
              "parent_frame_id": "lidar",  # You can replace this with your actual frame of reference
              "child_frame_id": "radar",
              "translation": {"x" : 0, "y" : 0, "z" : 0},
              "rotation": {"x" : 0, "y" : 0, "z" : -0.7071068, "w" : 0.7071068}  
            }
    
    writer.add_message(
                    channel_id=frame_idd,
                    log_time=timestamp_ns,
                    data=json.dumps(tf_flir_lidar).encode('utf-8'),
                    publish_time=timestamp_ns,
                )
    
    writer.add_message(
                    channel_id=frame_idd_car,
                    log_time=timestamp_ns,
                    data=json.dumps(tf_flir_car).encode('utf-8'),
                    publish_time=timestamp_ns,
                )
    
    writer.add_message(
                    channel_id=frame_idd_radar,
                    log_time=timestamp_ns,
                    data=json.dumps(tf_lidar_radar).encode('utf-8'),
                    publish_time=timestamp_ns,
                )
    
    # ================== 3D Boxes ==================
    box3d = sbgekinox_bcc['box3d-ensemble']
    centers = box3d[index].get_centers()
    dimensions = box3d[index].get_dimensions()
    rotations = box3d[index].get_rotations() 
    categories = box3d[index].get_categories() 
    
    rot = R.from_euler('xyz', rotations, degrees=True)

    # Convert to quaternions and print
    rot_quat = rot.as_quat() 
    
    cubes = []
    texts = []
    for index in range(len(centers)):
      pose = {
        "x" : np.float64(centers[index][0]), 
        "y" : np.float64(centers[index][1]), 
        "z" : np.float64(centers[index][2]), 
      }
      
      orient = {
        "x" : np.float64(rot_quat[index][0]), 
        "y" : np.float64(rot_quat[index][1]), 
        "z" : np.float64(rot_quat[index][2]), 
        "w" : np.float64(rot_quat[index][3]), 
      }
      
      size = {
        "x" : np.float64(dimensions[index][0]),
        "y" : np.float64(dimensions[index][1]),
        "z" : np.float64(dimensions[index][2])
      }
      
      color = [colors_dict.get(categories[index], {"r": 0.0, "g": 1.0, "b": 0.0, "a": 0.0}) ]
      
      
      cubes_message = {
        "pose" : {"position" : pose, "orientation" : orient },
        "size" : size, 
        "color" : color
      }
      
      text_message = {
        "pose" : {"position" : pose, "orientation" : orient },
        "billboard" : True,
        "font_size" : 0.5,
        "scale_invariant" : False,
        "color" : color,
        "text" : str(categories[index])
      }  
      
      cubes.append(cubes_message)
      texts.append(text_message)

    entity_message = {
      "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
      "frame_id": "world",  # You can replace this with your actual frame of reference
      "id" : "frame",
      "lifetime" : {"sec": 1, "nsec": 0},
      "frame_locked" : False,
      "cubes" : cubes,
      "texts" : texts
    }
    
    entities.append(entity_message)
    
    scene = {
      
      "entities" : entities,
    }
    
  
    writer.add_message(
                      channel_id=entity_channel_id,
                      log_time=timestamp_ns,
                      data=json.dumps(scene).encode('utf-8'),
                      publish_time=timestamp_ns,
                    ) 
    
    # ================== Location ==================
    location = sync[index]['sbgekinox_bcc_gps1pos'].raw.item()
    
    latitude = location[3]
    longitude = location[4]
    altitude = location[5]
    
    location_message = {
      "timestamp": {"sec": timestamp_ns, "nsec": timestep_nsec},
      "frame" : "map",
      "latitude" : latitude,
      "longitude" : longitude,
      "altitude" : altitude
    }
    
    writer.add_message(
                      channel_id=gps_channel_id,
                      log_time=timestamp_ns,
                      data=json.dumps(location_message).encode('utf-8'),
                      publish_time=timestamp_ns,
                    ) 
 
    # ================== Image annotations ==================
    annotations = camera_center_cyl['poly2d-detectron-undistort'][index].raw['data']
    timestamp = camera_center_cyl['poly2d-detectron-undistort'][index].raw["timestamp"]*1000
    delay_ms = -300
    timestamp_ns = int(timestamp) + delay_ms * 1000000
    timestep_sec = timestamp_ns // 1_000_000_000 
    timestep_nsec = timestamp_ns % 1_000_000_000
    points = []
    for annotation in annotations:
      polygon = annotation[0] 
      point_annotation = {
                "timestamp": {"sec": timestep_sec, "nsec": timestep_nsec},
                "type" : 3,
                "points" : [{"x": np.float64(point[0]), "y": np.float64(point[1])} for point in polygon],
                #"outline_color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.5},
                "outline_colors":[colors_dict.get(point[0], {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0})  for point in polygon],
                "fill_color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.01},
                "thickness" : 5,
              }
      #print(json.dumps(point_annotation,indent=2))
      points.append(point_annotation)
      
    annotation_message = {
                
                "points" : points
            }
    
    writer.add_message(
                      channel_id=annotation_id,
                      log_time=timestamp_ns,
                      data=json.dumps(annotation_message).encode('utf-8'),
                      publish_time=timestamp_ns,
                    ) 
    print(index)
    
writer.finish()
  