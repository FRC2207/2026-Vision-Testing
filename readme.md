# 2026 Vision Testing

So everything needs a little cleanup and stuff but the simplfied folder should be the files that just sends the detected fuel data points over network tables. The complex folder does a lot more like plot the path and sends the path over network tables.

## Files

- **video_feed_broadcaster.py**: Sets up a custom Flask app to make a website with the camera livestream
- **yolo_with_cam.py**: Runs the set YOLO vision model on a camera (defualt id is 0)
- **pt_to_wtv.py**: Lets you convert a .pt file to other file types for optimization (.onnx, .xml, etc)
- **map_maker.py**: Opens a interactive window to let you create a fuel map
- **map_creator.py**: Uses the vision model and some math or wtv to make a birdseye veiw of the image file (in Images directory)
- **camera_calibration.py**: This file is used for calculating the constants needed for focal length smth. Run the file, enter the paramters, hold the fuel a certain distance away and enter the other paramters then it will give you some values. Paste that into focal_calibration_data.txt for averaging. Average the "known_calibration_pixel_height" value and make sure when creating the camera you set the correct distance
- **livestream_reader.py**: Create a Camera object and reads form a livestream
- **onnx_to_rknn.py**: This file almost made me break my comptuer in frustration. Took 2 hours to setup but finally I got a .rknn file from my linux server, I just hope I set the paramters correct.

## Folders

### COMPLEX

The complex folder does a lot more then simplified. It finds the fuel and makes the path. This might be used idk

#### Classes

- **Camera.py**: Lets you setup a robot camera with lots and lots of paramters
- **CustomDBScan.py**: Setups a custom dbscan for filtering of points
- **NetworkTableHandler.py**: Lets you send data over network tables and stuff
- **PathPlanner.py**: Main file to get points and stuff. This is way smaller then the complex path planner file

#### Other Files

- **game_loop.py**: Lets you run one camera for the game loop, works (probaly)

### SIMPLIFIED

This folder is more simple, it just send the fuel positions over network tables it shuold be better for lag and stuff

#### Classes

- **Camera.py**: Lets you setup a robot camera with lots and lots of paramters
- **CustomDBScan.py**: Setups a custom dbscan for filtering of points
- **Fuel.py**: Fuel object thingie
- **FuelTracker.py**: Simple object but later can be used for more advanced math
- **NetworkTableHandler.py**: Lets you send data over network tables and stuff
- **PathPlanner.py**: Main file to get points and stuff. This is way smaller then the complex path planner file

#### Other Files

- **constants.py**: Lets you setup constasnts
- **game_loop.py**: Lets you "hopefully" run multiple camera's (hasn't been tested yet, theoretical)
- **solo_game_loop.py**: Lets you run one camera for the game loop, works (probaly)

### PLOTTERS

Bunch of testing files.

#### Files

- **astar_tech_with_tim.py**: A* algorithm from a really good programming youtuber resource (tech with tim)
- **b_spline_dbscan.py**: Implementation of interpolating spline and dbscan from ball_layout.json
- **b_spline.py**: Implementatoin of interpolating spline from ball_layout.py
- **grid.py**: Failed experiment
- **kochanek_bartels.py**: Another failed experiment

### YOLO_MODELS

Contains all the yolo vision models and other file (like .onnx) should be generally sorted by yolo_model and model version but some of them aren't sorted cause I got lazy

## Orange Pi 5 Info

- **IP**: 10.22.7.200
- **Username**: ubuntu
- **Password**: 2207vision
- **Auto-scheduler-function**: crontab -e
- **Clone-Repo**: git clone git@github.com:FRC2207/2026-Vision-Testing.git