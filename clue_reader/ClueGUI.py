#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import rospkg

import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np

from ImageProcessor import ImageProcessor
from CNNModel import CNNModelFast

import torch
import torch.nn as nn

import os
import string
from collections import defaultdict

class ClueGUI(QtWidgets.QMainWindow):

	def __init__(self):
		super(ClueGUI, self).__init__()
		loadUi("./ClueGUI.ui", self)

		self.bridge = CvBridge()
		rospy.sleep(1)

		rospy.init_node("ClueGUI", anonymous=True)

		self.debug = False

		self.cv_image_1 = None
		self.cv_image_2 = None
		self.cv_image_3 = None
		self.cv_image = None

		rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback_1, queue_size=1)
		rospy.Subscriber('/B1/rrbot/camera2/image_raw', Image, self.image_callback_2, queue_size=1)
		self.score_tracker_pub = rospy.Publisher('/score_tracker', String, queue_size=10)
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 

		self.clue_count = 0
		self.clue_count_pub = rospy.Publisher('/clue_count', String, queue_size=10)
		


		self.ImageProcessor = ImageProcessor()
		

		model_path = os.path.join(package_path, 'clue_reader')

		self.CNNModel = CNNModelFast()
		model_file = os.path.join(model_path, 'test_cnn_fast.pth')
		if not os.path.exists(model_file):
			raise FileNotFoundError(f"Model file not found: {model_file}")
		self.CNNModel.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
		self.label_map = list(string.ascii_uppercase + string.digits)

		self.team_name = "Weymen"
		self.password = "Koo"
		self.current_guess_counter = defaultdict(lambda: 0)
		self.sent_contexts = []
		self.possible_contexts = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]
		self.consecutive_empty = 10

		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.image_callback)
		self.timer.start(20)  # Publish at 10 Hz
	
		self.start_timer_button.clicked.connect(self.start_timer)
		self.stop_timer_button.clicked.connect(self.stop_timer)
		self.restart_button.clicked.connect(self.restart)

	


	# Converts the OpenCV frame to QPixmap for display
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
							 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	#Takes opencv image
	def resize_frame(self, label, frame):
		label_width, label_height = label.width(), label.height()
		frame_height, frame_width = frame.shape[:2]

		# Scale factor while maintaining aspect ratio
		scale = min(label_width / frame_width, label_height / frame_height)
		new_width, new_height = int(frame_width * scale), int(frame_height * scale)

		return cv2.resize(frame, (new_width, new_height))

	#Takes opencv frame
	def update_image_label(self, label, frame):
		resized_frame = self.resize_frame(label, frame)
		label.setPixmap(self.convert_cv_to_pixmap(resized_frame))


	def image_callback_1(self, msg):
		try:
			self.cv_image_1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			self.cv_image_1 = self.cv_image_1[200:-100]  # Crop out sky and ground
			if self.debug:
				self.update_image_label(self.camera1_label, self.cv_image_1)
		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")

	def image_callback_2(self, msg):
		try:
			self.cv_image_2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			self.cv_image_2 = self.cv_image_2[200:-100]  # Crop out sky and ground
			if self.debug:
				self.update_image_label(self.camera2_label, self.cv_image_2)
		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")


	def compare_and_choose_best_image(self):
		# Compare the clueboard sizes from each camera
		max_area = 0
		best_image = None
		camera = None

		# Check camera 1
		if self.cv_image_1 is not None:
			# masked_image_1 = self.ImageProcessor.threshold_blue(self.cv_image_1, hl=0, hh=10, sl=0, sh=10, vl=80, vh=220)
			area_1 = self.ImageProcessor.biggest_blue(self.cv_image_1)
			if area_1 > max_area:
				max_area = area_1
				best_image = self.cv_image_1
				camera=1

		# Check camera 2
		if self.cv_image_2 is not None:
			# masked_image_2 = self.ImageProcessor.threshold_blue(self.cv_image_2, hl=0, hh=10, sl=0, sh=10, vl=80, vh=220)
			area_2 = self.ImageProcessor.biggest_blue(self.cv_image_2)
			if area_2 > max_area:
				max_area = area_2
				best_image = self.cv_image_2
				camera = 2

		#print(f"max area: {max_area}, camera: {camera}")
		if best_image is not None and self.debug:
			self.update_image_label(self.best_image_label, best_image)

		return best_image



	def image_callback(self):
		try:
			if self.cv_image_1 is None or self.cv_image_2 is None:
				rospy.logwarn("Waiting for all camera images to initialize...")
			else:
				# dynamically choose image with the largest clueboard
				self.cv_image = self.compare_and_choose_best_image()

			if self.cv_image is None:
				rospy.logwarn("No valid clueboard detected in any camera.")
				return

			largest_area = self.ImageProcessor.biggest_blue(self.cv_image)
		
			top_output = ""
			bottom_output = ""

			if largest_area > 10000: 
				clue_board = self.ImageProcessor.rect_and_detect(self.cv_image)
				if clue_board is not None:
					self.update_image_label(self.clue_board_label, clue_board)
					subimages = self.ImageProcessor.get_subimages(clue_board)
					
					#Context
					if (subimages[0] != None):
						top_stacked_image = self.stack_images_vertically(subimages[0])
						if self.debug:
							self.update_image_label(self.top_subimages_label, top_stacked_image)
						top_output = self.predict_word(subimages[0])
						self.top_pred_label.setText(top_output)

					#Clue
					if (subimages[1] != None):
						bottom_stacked_image = self.stack_images_vertically(subimages[1])
						if self.debug:
							self.update_image_label(self.bottom_subimages_label, bottom_stacked_image)
						bottom_output = self.predict_word(subimages[1])
						self.bottom_pred_label.setText(bottom_output)
					
				print(f"{top_output}, {bottom_output}")
				if top_output and bottom_output:
					self.consecutive_empty = 0
					if top_output in self.possible_contexts and top_output not in self.sent_contexts: #If context is correct
						guess = f"{top_output},{bottom_output}"
						self.current_guess_counter[guess] += 1
				if  self.current_guess_counter and max(self.current_guess_counter.items(), key=lambda x: x[1])[1] > 30:
					self.send_guess()

	
			else:
				self.consecutive_empty +=1
				if self.consecutive_empty == 5: #once we've seen enough empty frames, send best guess
					if self.current_guess_counter:
						self.send_guess()
						self.clue_count += 1
						self.clue_count_pub.publish(str(self.clue_count))

			#print(self.consecutive_empty)
			self.update_guess_list()
		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")

	def send_guess(self):
		guess = max(self.current_guess_counter.items(), key=lambda x: x[1])[0]
		context, clue = guess.split(',')
		context_index = self.possible_contexts.index(context)
		self.sent_contexts.append(context)
		message = f'{self.team_name},{self.password},{context_index+1},{clue}'
		rospy.loginfo(f"Publishing to /score_tracker: {message}")
		self.score_tracker_pub.publish(message)
		rospy.sleep(1)
		self.current_guess_counter = defaultdict(lambda: 0)


	def predict_word(self,letter_images):
		predicted_word = ""
		if letter_images != None and len(letter_images) != 0:
			with torch.no_grad():
				batch = torch.stack([torch.from_numpy(img).float() for img in letter_images])
				outputs = self.CNNModel(batch)
				_, predicted_labels = torch.max(outputs, 1)
				predicted_word = ''.join([self.label_map[label.item()] for label in predicted_labels])
		return predicted_word

	def update_guess_list(self):
		"""
		Updates the current_guesses_list widget to display guesses from current_guess_counter,
		sorted by the highest count.
		"""
		self.current_guesses_widget.clear()  # Clear existing items in the list widget

		# Sort items by count in descending order
		sorted_guesses = sorted(self.current_guess_counter.items(), key=lambda item: item[1], reverse=True)

		# Add sorted guesses to the list widget
		for guess, count in sorted_guesses:
			display_text = f"{guess}: {count}"  # Format guess and its count
			self.current_guesses_widget.addItem(display_text)

	def stack_images_vertically(self, image_list):
		# Ensure all images are the same width
		widths = [img.shape[1] for img in image_list]
		max_width = max(widths)

		# Resize images to the same width while maintaining aspect ratio
		resized_images = []
		for img in image_list:
			height, width = img.shape[:2]
			if width != 0:
				new_height = int(height * (max_width / width))
				resized_img = cv2.resize(img, (max_width, new_height))
				resized_images.append(resized_img)

		# Stack images vertically
		stacked_image = np.vstack(resized_images)
		return stacked_image
	
	def start_timer(self):
		message = f'{self.team_name},{self.password},0,NA'
		rospy.loginfo(f"Publishing to /score_tracker: {message}")
		self.score_tracker_pub.publish(message)
		rospy.sleep(1)

	def stop_timer(self):
		message = f'{self.team_name},{self.password},-1,NA'
		rospy.loginfo(f"Publishing to /score_tracker: {message}")
		self.score_tracker_pub.publish(message)
		rospy.sleep(1)

	def restart(self):
		self.current_guess_counter = defaultdict(lambda: 0)
		self.sent_contexts = []
		#self.consecutive_empty = 0

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = ClueGUI()
	myApp.show()
	sys.exit(app.exec_())