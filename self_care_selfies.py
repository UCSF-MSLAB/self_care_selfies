#!/usr/local/opt/python@3.9/bin/python3.9

import csv
import cv2
import math
import mediapipe as mp
import os
import sys

mpHands = mp.solutions.hands
mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose_drawing_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

SWITCH_HANDS = {
	"Left": "Right",
	"Right": "Left"
}

# landmarks for whole-body pose analysis
poseDict = {
	0: "nose",
	1: "left_eye_inner",
	2: "left_eye",
	3: "left_eye_outer",
	4: "right_eye_inner",
	5: "right_eye",
	6: "right_eye_outer",
	7: "left_ear",
	8: "right_ear",
	9: "mouth_left",
	10: "mouth_right",
	11: "left_shoulder",
	12: "right_shoulder",
	13: "left_elbow",
	14: "right_elbow",
	15: "left_wrist",
	16: "right_wrist",
	17: "left_pinky",
	18: "right_pinky",
	19: "left_index",
	20: "right_index",
	21: "left_thumb",
	22: "right_thumb",
	23: "left_hip",
	24: "right_hip",
	25: "left_knee",
	26: "right_knee",
	27: "left_ankle",
	28: "right_ankle",
	29: "left_heel",
	30: "right_heel",
	31: "left_foot_index",
	32: "right_foot_index"
}

# landmarks for hand analysis
handDict = {
	0: "wrist",
	1: "thumb_cmc",
	2: "thumb_mcp",
	3: "thumb_ip",
	4: "thumb_tip",
	5: "index_finger_mcp",
	6: "index_finger_pip",
	7: "index_finger_dip",
	8: "index_finger_tip",
	9: "middle_finger_mcp",
	10: "middle_finger_pip",
	11: "middle_finger_dip",
	12: "middle_finger_tip",
	13: "ring_finger_mcp",
	14: "ring_finger_pip",
	15: "ring_finger_dip",
	16: "ring_finger_tip",
	17: "pinky_mcp",
	18: "pinky_pip",
	19: "pinky_dip",
	20: "pinky_tip",
}

def count_peaks(landmarks, axis=1):
	# count changes in direction for an array of landmark x,y values
	num_peaks = 0
	heading_down = False
	for i, point1 in enumerate(landmarks[:-1]):
		cur = point1[axis]
		next = landmarks[i + 1][axis]
		if cur == next:
			continue
		if cur > next and not heading_down:
			num_peaks += 1
		heading_down = cur > next
	return num_peaks

def count_velocity_peaks(landmarks):
	# compute velocity for each pair of landmarks, then count changes
	velocity = []
	for i, point1 in enumerate(landmarks[:-1]):
		point2 = landmarks[i + 1]
		velocity.append((i, distance(point1, point2)))
	return count_peaks(velocity)

def distance(point1, point2):
	# compute distance between 2 points. these can be 2, 3 or any dimensions
	sum_of_squares = 0
	for dim, val in enumerate(point1):
		delta = val - point2[dim]
		sum_of_squares += delta*delta
	return math.sqrt(sum_of_squares)

def displacement(landmark):
	# find distance between first and last landmark points
	return distance(landmark[0], landmark[-1])

def total_travel(landmarks):
	# sum of distances between adjacent points the landmark travels
	travel = 0
	for i, point1 in enumerate(landmarks[:-1]):
		point2 = landmarks[i + 1]
		travel += distance(point1, point2)
	return travel

def average_velocity(landmarks):
	# computer average velocity (average of distance between adjacent images)
	return total_travel(landmarks) / len(landmarks)

def peak_velocity(landmarks):
	# find the max velocity (distance between adjacent images in video)
	peak_travel = 0
	for i, point1 in enumerate(landmarks[:-1]):
		point2 = landmarks[i + 1]
		travel = distance(point1, point2)
		if travel > peak_travel:
			peak_travel = travel
	return peak_travel

def hand_name(hand):
	# flip hand (right -> left, etc.)
	which_hand = hand.classification[0].label
	which_hand = SWITCH_HANDS[which_hand]
	return which_hand

def process_pose(frame_num, video_landmark_data, results, img):
	# given an image from a video, gather up landmark data for whole body Pose
	h, w, _ = img.shape
	if results.pose_landmarks:
		for idx, lm in enumerate(results.pose_landmarks.landmark):
			video_landmark_data["Pose"][idx].append((int(lm.x*w), int(lm.y*h)))
		mpDraw.draw_landmarks(img, results.pose_landmarks,
			mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=pose_drawing_style)

def process_hand(frame_num, video_landmark_data, results, img):
	# given an image from a video, gather up landmark data for hands
	if not results.multi_handedness:
		return []
	h, w, _ = img.shape
	the_landmarks = results.multi_hand_landmarks
	if the_landmarks:
		for hand, landmarks in enumerate(the_landmarks):
			mpDraw.draw_landmarks(img, landmarks, mpHands.HAND_CONNECTIONS)
			which_hand = hand_name(results.multi_handedness[hand])
			for idx, lm in enumerate(landmarks.landmark):
				video_landmark_data[which_hand][idx].append((int(lm.x*w), int(lm.y*h)))

def process_frame(video_landmark_data, frame_num, 
		processor, result_processor, img):
	# process one frame from a video
	# put results in video_landmark_data
	# use processor to compute landmarks, and result_processor to store results
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = processor.process(imgRGB)
	result_processor(frame_num, video_landmark_data, results, img)
	cv2.imshow("Image", img)
	cv2.waitKey(1)

def process_video(video, processor, result_processor, 
		start_time_secs, end_time_secs):
	# process an entire video, given the single image 'processor'
	# and the method to store theresults
	video_data = {
		"Left": [],
		"Right": [],
		"Pose": []
	}
	for landmark_index in range(33):
		video_data["Left"].append([])
		video_data["Right"].append([])
		video_data["Pose"].append([])
	fps = video.get(cv2.CAP_PROP_FPS )
	while True:
		success, img = video.read()
		if not success:
			break
		currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
		# currentSec = video.get(cv2.CAP_PROP_POS_MSEC)
		currentSec = (currentFrame * fps) / 1000.0
		if currentSec <= start_time_secs:
			continue
		if currentSec > end_time_secs:
			break
		process_frame(video_data, currentFrame, 
			processor, result_processor, img)
	return video_data

def get_video(participant, activity_date, activity_video, ext, video_dir):
	# read the video at the path indicated by participant, activity_date,
	# and name of activity_video/extension
	video_file_path = os.path.join(video_dir, participant, activity_date, 
		"{}.{}".format(activity_video, ext))
	return cv2.VideoCapture(video_file_path)

def compute_metrics(video_data, hand, feature):
	# given analyzed video information (video_data), compute the set of
	# setrics
	if not video_data["data"][hand][feature]:
		return None
	landmarks = video_data["data"][hand][feature]
	metrics = {
		"displacement": displacement(landmarks),
		"total_travel": total_travel(landmarks),
		"average_velocity": average_velocity(landmarks),
		"peak_velocity": peak_velocity(landmarks),
		"normed_velocity": average_velocity(landmarks) / peak_velocity(landmarks),
		"velocity_peaks": count_velocity_peaks(landmarks) / len(landmarks)
	}
	return metrics

def choose_video_type(activity):
	# use filename to select processor method, result_processor method
	# names of landmarks, and which features to compute
	activity_lower = activity.lower()
	if activity_lower.startswith("gait"):
		processor = mp_pose.Pose(min_detection_confidence=0.5, 
			min_tracking_confidence=0.5)
		result_processor = process_pose
		lm_dict = poseDict
		# select which features to process for Gait videos
		# left foot (31) and right foot (32)
		features = [31, 32]
	elif activity_lower.startswith("talk"):
		processor = mp_pose.Pose(min_detection_confidence=0.5, 
			min_tracking_confidence=0.5)
		result_processor = process_pose
		lm_dict = poseDict
		# select which features to process for Talk videos
		# left/right eyes (1, 5) and left/right mouth (9, 10)
		features = [1, 5, 9, 10]
	else:
		processor = mpHands.Hands(static_image_mode=False,
			max_num_hands=2,
			min_detection_confidence=DETECTION_CONFIDENCE,
			min_tracking_confidence=TRACKING_CONFIDENCE)
		result_processor = process_hand
		lm_dict = handDict
		# select which features to process for Hand videos
		# wrist (0) and index finger tip (8)
		features = [0, 8]
	return processor, result_processor, lm_dict, features

def process_video_for_participant_date_activity(
		participant, activity_date, activity, ext,
		start_time_secs, end_time_secs, video_dir):
	# read video and process it, returning the metrics
	print("processing {} performing {} on {}".format(
		participant, activity, activity_date))
	processor, result_processor, lm_dict, features = choose_video_type(activity)
	video = get_video(participant, activity_date, activity, ext, video_dir)
	video_data = {
		"participant": participant,
		"activity_date": activity_date,
		"activity": activity,
		"lm_dict": lm_dict,
		"features": features,
		"data": process_video(video, processor, result_processor,
			start_time_secs, end_time_secs)
	}
	return video_data

def get_video_data_using_csv(csv_file, video_dir):
	# read csv that specifies which videos to process, and process them
	rows = []
	with open(csv_file, 'r', encoding='utf-8-sig') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			rows.append(row)
	all_video_data = []
	headers = ["activity", "hand", "landmark", "participant", "date", "displacement",
		"total_travel", "average_velocity", 
		"peak_velocity", "normed_velocity", "velocity_peaks"]
	output_rows = []
	for row in rows[1:]:
		participant, activity_date, activity, ext = row[:4]
		start_time_secs, end_time_secs = [float(v) for v in row[4:6]]
		video_data = process_video_for_participant_date_activity(
			participant, activity_date, activity, ext,
			start_time_secs, end_time_secs, video_dir)
		all_video_data.append(video_data)
	return all_video_data

def ignore_key(participant, activity_date, activity):
	# compute key for video so we can tell if we've already processed it
	return "{}-{}-{}".format(participant, activity_date, activity)

def get_video_data_crawling_dir(video_dir, ignore_set):
	# crawl video directory and process videos not yet dealt with before
	all_video_data = []
	for (dirpath, dirnames, filenames) in os.walk(video_dir):
		for filename in filenames:
			name, ext = os.path.splitext(filename)
			ext = ext.lower()[1:]
			if not (ext == "mov" or ext == "mp4"):
				continue
			parents = dirpath.split("/")
			participant, activity_date = parents[-2:]
			key = ignore_key(participant, activity_date, name)
			if key in ignore_set:
				continue
			video_data = process_video_for_participant_date_activity(
				participant, activity_date, name, ext,
				0, 24*60*60, video_dir)
			all_video_data.append(video_data)
	return all_video_data

def read_ignore_set(output_file):
	# read the output data so we can tell which videos we've already processed
	ignore_set = set()
	rows = []
	if os.path.exists(output_file):
		with open(output_file, 'r', encoding='utf-8-sig') as csvfile:
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				rows.append(row)
			rows = rows[1:]
			for row in rows:
				participant = row[3]
				activity_date = row[4]
				activity = row[0]
				key = ignore_key(participant, activity_date, activity)
				ignore_set.add(key)
	return ignore_set, rows

def process_all_videos(csv_file, video_dir, output_file):
	# process all videos. If csv_file is provided, use that, otherwise
	# crawl video_dir
	output_rows = []
	if csv_file:
		all_video_data = get_video_data_using_csv(csv_file, video_dir)
	else:
		ignore_set, output_rows = read_ignore_set(output_file)
		all_video_data = get_video_data_crawling_dir(video_dir, ignore_set)
	headers = ["activity", "hand", "landmark", "participant", "date", "displacement",
		"total_travel", "average_velocity", 
		"peak_velocity", "normed_velocity", "velocity_peaks"]
	for one_video_data in all_video_data:
		for hand in ["Left", "Right", "Pose"]:
			for landmark in one_video_data["features"]:
				metrics = compute_metrics(one_video_data, hand, landmark)
				if metrics:
					output_rows.append([
						one_video_data["activity"],
						hand, 
						landmark_to_name(one_video_data["lm_dict"], landmark),
						one_video_data["participant"],
						one_video_data["activity_date"],
						metrics["displacement"], 
						metrics["total_travel"], 
						metrics["average_velocity"], 
						metrics["peak_velocity"], 
						metrics["normed_velocity"], 
						metrics["velocity_peaks"] 
						])
	output_rows.sort(key=lambda row: "{}={}={}={}={}".format(
		row[0], row[1], row[2], row[3], row[4]))
	output_rows.insert(0, headers)
	with open(output_file, 'w', newline='') as csvfile:
		my_writer = csv.writer(csvfile, delimiter=',',)
		my_writer.writerows(output_rows)


def arg(args, i, default_value):
	# get value of the ith command line arg, with a default if missing
	if i < len(args):
		return args[i]
	return default_value

def name_to_landmark(lm_dict, name):
	# given a landmark name dictionary, look up landmark id for the given name
	for landmark in lm_dict.keys():
		if lm_dict[landmark] == name:
			return landmark
	return None

def landmark_to_name(lm_dict, landmark):
	# given a landmark name directory, get name for landmark
	if landmark not in lm_dict:
		return None
	return lm_dict[landmark]

def main():
	# main program that reads command-line arguments and processes videos
	args = sys.argv[1:]
	if len(args) == 1 and args[0] == "--help":
		print("usage: python self_care_selfies.py [<video dir = 'videos'> [<output csv file = 'output.csv'> [<input csv file>]]]")
		exit()
	video_dir = arg(args, 0, "videos")
	output_file = arg(args, 1, "output.csv")
	csv_file = arg(args, 2, "")
	process_all_videos(csv_file, video_dir, output_file)

if __name__ == "__main__":
	main()
