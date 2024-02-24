# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
import os

import torch
import utils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

imgfilepath = '/Users/sanikabharvirkar/Downloads/perception/yolov5/runs/detect/exp2/conesimage.png'
coords = '/Users/sanikabharvirkar/Downloads/perception/yolov5/runs/detect/exp2/labels/conesimage.txt'

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', '/Users/sanikabharvirkar/Downloads/perception/yolov5/best.pt')  # custom trained model

# Images
im = '/Users/sanikabharvirkar/Downloads/perception/yolov5/runs/detect/exp2/cones2.png'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

frame_count = 0
def bounding_boxes(s): 
	image = cv2.imread(s)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([20, 100, 100])
	upper_yellow = np.array([30, 255, 255])

	# Create a mask to filter out pixels that are not in the yellow color range
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	# Define the lower and upper bounds for the black color in BGR
	lower_black = np.array([0, 0, 0])
	upper_black = np.array([50, 50, 50])

	# Create a mask for black pixels
	mask_black = cv2.inRange(image, lower_black, upper_black)

	# Combine the masks to get a mask for both yellow and black pixels
	combined_mask = cv2.bitwise_or(mask_yellow, mask_black)

	# Find contours in the combined mask
	contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Draw bounding boxes around each yellow object
	boxes = []
	result_image = image.copy()
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		min_area_threshold = 4000
		if w * h > min_area_threshold:
			boxes.append((x, y, x + w, y + h))
			cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness

	# Display the result
	cv2.imshow('Bounding Boxes around Yellow Objects', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return boxes

def find_marker(image, coords):

	results.print()
	print(results.xyxy[0])
	print(results.pandas().xyxy[0])
	# convert the image to grayscale, blur it, and detect edges
	img = cv2.imread(image)

	original_height, original_width = img.shape[:2]
	print(img.shape)
	file_content = ""
	with open(coords, 'r') as file:
		file_content = file.read()
	nums = [float(num) for num in file_content.split()] 
	if (len(nums) < 5): 
		print("error getting getting coords from text")
		return
	label = nums[0]
	xcoord = int(nums[1] * original_width)
	ycoord = int(nums[2] * original_height)
	width = int(nums[3] * original_width)
	height = int(nums[4] * original_height)

	boundedimg = img[ycoord:ycoord + height, xcoord:xcoord + width]
	#cv2.imshow('Cropped Image', boundedimg)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()



	print(coords)

def cropbox(image): 
	img = cv2.imread(image)
	coords = results.pandas().xyxy[0]
	position = coords.iloc[0][['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].values
	print("position", position)
	actual_coords = [int(num) for num in position[:5]]
	
	boundedimg = img[int(position[1]):int(position[3]), int(position[0]):int(position[2])]
	return boundedimg

def pixelCount(image): 
	  # Create a mask using the specified color range
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	lower_yellow = np.array([20, 100, 100])
	upper_yellow = np.array([30, 255, 255])
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

	lower_black = np.array([0, 0, 0])
	upper_black = np.array([50, 50, 50])
	mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Count the number of non-zero pixels in the mask
	pixel_count = np.count_nonzero(mask_black) + np.count_nonzero(mask_yellow)
	return pixel_count
	
def drawContour(image): 

	# Convert the image from BGR to HSV
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# Define the lower and upper bounds for the yellow color in HSV
	lower_yellow = np.array([25, 105, 105])
	upper_yellow = np.array([30, 255, 255])

	# Define the lower and upper bounds for the black color in BGR
	lower_black = np.array([0, 0, 0])
	upper_black = np.array([50, 50, 50])

	# Create masks to filter out pixels that are not in the yellow and black color ranges
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	mask_black = cv2.inRange(image, lower_black, upper_black)

	# Combine the masks to get a mask for both yellow and black pixels
	combined_mask = cv2.bitwise_or(mask_yellow, mask_black)

	# Find contours in the combined mask
	contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	frame_with_contours = image.copy()
	cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)  # -1 means draw all contours

	
	cv2.imshow('Cropped Image', frame_with_contours)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Find the contour with the maximum area (assumed to be the yellow and black region)
	max_contour = max(contours, key=cv2.contourArea)

	# Get the minimum bounding box around the contour
	#rect = cv2.minAreaRect(max_contour)
	#box = cv2.boxPoints(rect)
	#box = np.int0(box)

	# Get the straight bounding rectangle around the contour
	x, y, w, h = cv2.boundingRect(max_contour)

	# Draw the bounding box on the original image
	result_image = image.copy()
	cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness

	# Draw the minimum bounding box on the original image
	#result_image = image.copy()
	#cv2.drawContours(result_image, [box], 0, (0, 255, 0), 2)  # green color for the bounding box

	cv2.imshow('resulting crop', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	#return pixel_counts
	return (x, y, x + w, y + h)

def distance_to_camera(real, focal, obj, img, sensor):
	# compute and return the distance from the maker to the camera
	return (focal * real * img) / (obj * sensor)


def find_distances(image, frame_count, timer, focalLength): 
	img = cv2.imread(image)
	results = model(image)
	
	s = None
	while results.pandas().xyxy is None or len(results.pandas().xyxy[0]) < 2:
		s, frame_count = extract_frames(timer, frame_count)
		img = cv2.imread(s)
		results = model(s)
		timer += 0.1
	
	bounding_boxes(s)
	img_height = img.shape[0]
	for row in results.pandas().xyxy[0].iloc: 
		position = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].values
		print("position", position)
		actual_coords = [int(num) for num in position[:5]]


		# Draw the bounding box on the image
		cv2.rectangle(img, (int(position[0]), int(position[1])), (int(position[2]), int(position[3])), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness

		# Display the result
		cv2.imshow('Bounding Box', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		padding = 25
		boundedimg = img[int(position[1])-padding:int(position[3])+padding, int(position[0])-padding:int(position[2])+padding]
		cv2.imshow('Cropped Image', boundedimg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		#contoured = pixelCount(boundedimg)
		contoured = drawContour(boundedimg)
		print("rect", contoured)
		height = contoured[0][1] - contoured[1][1]
		print("pixels", contoured)
		if focalLength is None: 
			focalLength = (height * 91 * 5) / (12 * img_height)
		distance = distance_to_camera(12, focalLength, height, img_height, 5)
		print("distance", distance)

def find_distances_naive(image, focalLength): 
	
	img = cv2.imread(image)
	boxes = bounding_boxes(image)
	img_height = img.shape[0]
	
	format_boxes = []
	for position in boxes: 
		
		# Draw the bounding box on the image
		cv2.rectangle(img, (int(position[0]), int(position[1])), (int(position[2]), int(position[3])), (0, 255, 0), 2)  # (0, 255, 0) is the color (green), 2 is the thickness

		# Display the result
		cv2.imshow('Bounding Box', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		padding = 25
		boundedimg = img[int(position[1])-padding:int(position[3])+padding, int(position[0])-padding:int(position[2])+padding]
		cv2.imshow('Cropped Image', boundedimg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		#contoured = pixelCount(boundedimg)
		contoured = drawContour(boundedimg)
	
		height = contoured[3] - contoured[1]
		print(height)
		if focalLength is None: 
			focalLength = (height * 69 * 5) / (12 * img_height)
		distance = distance_to_camera(12, focalLength, height, img_height, 5)
		print("distance", distance)
		format_boxes.append([(position[0], position[1], position[2], position[3]), distance])
	return format_boxes

def extract_frames(seconds, frame_count): 
	video = cv2.VideoCapture('/Users/sanikabharvirkar/Downloads/perception/yolov5/videos/cones1.mp4')
	fps = video.get(cv2.CAP_PROP_FPS)
	frame_id = int(fps*(seconds))
	video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
	ret, frame = video.read()
	if ret: 
		s = '/Users/sanikabharvirkar/Downloads/perception/yolov5/frames'+ '/frame_' + str(frame_count) + '.jpg'
		frame_path = os.path.join(s)
		cv2.imwrite(frame_path, frame)
		frame_count = frame_count + 1
		return s, frame_count

def format_output(s, cone_distances): 
	# Read the image
	image = cv2.imread(s)
	result_image = image.copy()
	for cone in cone_distances: 

		cv2.rectangle(result_image, (cone[0][0], cone[0][1]), (cone[0][2], cone[0][3]), (180, 105, 255), 2)  # (0, 255, 0) is the color (green), 2 is the thickness
		note = "{:.2f}".format(cone[1]) + "in"
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 1
		font_thickness = 4
		text_size = cv2.getTextSize(note, font, font_scale, font_thickness)[0]
		text_position = (cone[0][2] - text_size[0], cone[0][3] + text_size[1])
		cv2.putText(result_image, note, text_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

	# Display the result
	cv2.imshow('Bounding Box with Note', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	
#find_marker(imgfilepath, coords)
#cropped = cropbox(imgfilepath)
#marker = drawContour(cropped)
timer = 0 
frame_count = 0
s, frame_count = extract_frames(timer, frame_count)
focalLength = None #(marker[1][0] * 104) / 13.75
#print("focallength", focalLength)
#find_distances(s, frame_count, timer, focalLength)
cone_distances = find_distances_naive(s, focalLength)
format_output(s, cone_distances)


	



