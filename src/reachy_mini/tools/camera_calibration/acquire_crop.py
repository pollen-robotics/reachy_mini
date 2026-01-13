import os
import time

import cv2

from reachy_mini import ReachyMini
from reachy_mini.media.camera_constants import CameraResolution

save_path = "./images"
os.makedirs(save_path, exist_ok=True)

cv2.namedWindow("Reachy Mini Camera")
with ReachyMini(media_backend="gstreamer") as reachy_mini:
    available_resolutions = reachy_mini.media.camera.camera_specs.available_resolutions
    
    max_resolution = None
    max_resolution_value = 0
    for resolution_enum in available_resolutions:
        res = resolution_enum.value[:2]
        
        if res[0] * res[1] > max_resolution_value:
            max_resolution_value = res[0] * res[1]
            max_resolution = resolution_enum
            
            
    current_res = max_resolution
    reachy_mini.media.camera.close()
    reachy_mini.media.camera.set_resolution(current_res)
    reachy_mini.media.camera.open()
    
    remaining_resolutions = []
    for av in available_resolutions:
        if av == max_resolution:
            continue
        remaining_resolutions.append(av)
    
    time.sleep(2)
    try:
        i = -1
        while True:
            frame = reachy_mini.media.get_frame()
            if frame is None:
                print("Failed to grab frame.")
                continue
                
            cv2.imshow("Reachy Mini Camera", frame)
            # cv2.imshow("Reachy Mini Camera", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
            key = cv2.waitKey(1)
            if key == 13:
                print(f"Current resolution {current_res}")
                image_save_path = os.path.join(save_path, f"{current_res}.png")
                cv2.imwrite(image_save_path, frame)
                print(f"Saving {image_save_path}")
                i += 1
                if i < len(remaining_resolutions):
                    current_res = remaining_resolutions[i]
                    reachy_mini.media.camera.close()
                    reachy_mini.media.camera.set_resolution(current_res)
                    reachy_mini.media.camera.open()
                else:
                    break
            time.sleep(1.0 / 30)
                                                                                                                                                                                                                
    except KeyboardInterrupt:
        print("Interrupted, closing viewer.")
                                                                                                                                                                                                                                