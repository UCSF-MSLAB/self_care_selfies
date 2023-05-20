Self Care Selfies helps MS and other patients gain more insight into their health situation. This repository contains code that analyzes videos captured by the patient and calculates several metrics. These metrics have been shown to correlate well with clinical measures of disease status.

To use this code, place the videos to be analyzed into a directory structure like this:

videos
  patient1
    01-01-2023
      BrushL.mp4
      BrushR.mp4
      Talk.mp4
    01-08-2023
      BrushL.mp4
      BrushR.mp4
      Talk.mp4
  patient2
    01-01-2023
      EatL.mov
      Button.mov
      Gait.mov
    01-08-2023
      EatL.mov
      Button.mov
      Gait.mov

The name of the video file represents the activity being performed 
by the patient. There are several supported activity types.

* Gait - walking
* Talk - facial expressions while reading or describing
* Button - fingers and wrist while buttoning a button
* Eat - fingers and wrist while eating
* Brush - fingers and wrist while brushing teeth

Add the suffix L and R for button, eat, and brush activities to
represent the hand being used.

If you name your video directory 'videos' and are ok having the 
output being written to 'output.csv' then you can simply execute 
the self_care_selfies.py code.

This command line will show the usage:

$ self_care_selfies.py --help
usage: python self_care_selfies.py [<video dir = 'videos'> [<output csv file = 'output.csv'> [<input csv file>]]]

The output is a csv file with metrics for each video.

* activity - name of video
* hand - Left, Right or Pose
* landmark - name of landmark, such as left_foot_index
* participant - id of patient
* date - date video was taken
* displacement - distance from start to end of landmark position in frame
* total_travel - total distance traveled of landmark
* average_velocity - total_travel / num_frames in video
* peak_velocity - highest distance traveled between adjacent frames
* normed_velocity - average velocity / peak velocity
* velocity_peaks - number of peaks in velocity / num_frames in video


