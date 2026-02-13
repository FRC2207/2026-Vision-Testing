So everything needs a little cleanup and stuff but the simplfied folder should be the files that just sends the detected fuel data points over network tables.
The complex folder does a lot more like plot the path and sends the path over network tables.
To test path planning, you can run map_creator to create a map with vision from photos or map_maker to run a interactive thingie
To convert .pt files to wtv extension you want to make it faster run pt_to_wtv.py
video_feed_broadcaster is to test reading frames over wifi
camera_calibration will let you get all teh data needed for focal lenght calculations but i avereged several runs to get a better results