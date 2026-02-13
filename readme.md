So everything needs a little cleanup and stuff but the simplfied folder should be the files that just sends the detected fuel data points over network tables.
The complex folder does a lot more like plot the path and sends the path over network tables.

FILES:
 - video_feed_broadcaster.py: Sets up a custom Flask app to make a website with the camera livestream
 - yolo_with_cam.py: Runs the set YOLO vision model on a camera (defualt id is 0)
 - pt_to_wtv.py: Lets you convert a .pt file to other file types for optimization (.onnx, .xml, etc)
 - map_maker.py: Opens a interactive window to let you create a fuel map
 - map_creator.py: Uses the vision model and some math or wtv to make a birdseye veiw of the image file (in Images directory)
 - camera_calibration.py: This file is used for calculating the constants needed for focal length smth. Run the file, enter the paramters, hold the fuel a certain distance away and enter the other paramters then it will give you some values. Paste that into focal_calibration_data.txt for averaging. Average the "known_calibration_pixel_height" value and make sure when creating the camera you set the correct distance
 - livestream_reader.py: Create a Camera object and reads form a livestream

FOLDERS:
 - COMPLEX
  - The complex folder does a lot more then simplified. It finds the fuel and makes the path. This might be used idk
 - SIMPLIFIED
  - This folder is more simple, it just send the fuel positions over network tables it shuold be better for lag and stuff.
 - PLOTTERS
  - Bunch of testing files.
   - FILES
    - astar_tech_with_tim.py: A* algorithm from a really good programming youtuber resource (tech with tim)
    - b_spline_dbscan.py: Implementation of interpolating spline and dbscan from ball_layout.json
    - b_spline.py: Implementatoin of interpolating spline from ball_layout.py
    - grid.py: Failed experiment
    - kochanek_bartels.py: Another failed experiment
 - YOLO_MODELS
  - Contains all the yolo vision models and other file (like .onnx) should be generally sorted by yolo_model and model version but some of them aren't sorted cause I got lazy

ORANGE PI 5 INFO:
 - ip: 10.22.7.200
 - username: ubuntu
 - passowrd: 2207vision
 - auto-scheduler-function: crontab -e