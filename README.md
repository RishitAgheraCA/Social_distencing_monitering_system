# Social_distencing_monitering_system

##Detect the distance less than given distance among the persons detected in the video. 

1. clone the code.
2. download the Yolo3 weights,cfg,coco.names and put to data/model. or dowload from the drive https://drive.google.com/drive/folders/1_r5fCdvYyf39e-YSeWm5m430YPHFsYq1
3. run footage_from_video.py this will save last image frame to root directory as last_frame.jpg
4. run distance_to_pixel.py. saved image will pop up. using mouse pointer, click the width that you know. for example we know that width of 3 tiles are 6 feet. so click start to one tiles to end of third tile. you will see the pixel size. remember it.
5. now, in the code of person_detection.py change value of width_of_3_tiles to pixel size showed in last file execution.
6. run person_detection.py.