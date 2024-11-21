#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import rospy
from sensor_msgs.msg import Image
import rospkg

import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./ClueGUI.ui", self)

		rospy.init_node("ClueGUI", anonymous=True)
		rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

		self.bridge = CvBridge()

		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg')  # Replace 'controller_pkg' with your package name
		template_image_path = f"{package_path}/clue_reader/assets/clue_template.png"  # Adjust path as needed
		self.template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
		if self.template_image is None:
			rospy.logerr("Failed to load template image.")
		else:
			rospy.loginfo(f"Loaded template image from {template_image_path}")
			self.template_label.setPixmap(self.convert_cv_to_pixmap(self.resize_frame(self.template_image, self.template_label)))

	# Converts the OpenCV frame to QPixmap for display
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
							 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	#Takes opencv
	def resize_frame(self, frame, label):
		label_width, label_height = label.width(), label.height()
		frame_height, frame_width = frame.shape[:2]

		# Scale factor while maintaining aspect ratio
		scale = min(label_width / frame_width, label_height / frame_height)
		new_width, new_height = int(frame_width * scale), int(frame_height * scale)

		return cv2.resize(frame, (new_width, new_height))

	def image_callback(self, msg):
		try:
			# Convert the ROS Image message to an OpenCV image
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

			pixmap = self.convert_cv_to_pixmap(self.resize_frame(cv_image, self.camera_label))
			self.camera_label.setPixmap(pixmap)

			cropped = self.sift_and_shift(cv_image)

			if cropped is not None:
				self.cropped_label.setPixmap(self.convert_cv_to_pixmap(self.resize_frame(cropped, self.cropped_label)))

		except CvBridgeError as e:
			rospy.logwarn(f"Error converting ROS Image to OpenCV: {e}")

	def sift_and_shift(self, frame):
		self.template_image

		if self.template_image is None or frame is None:
			print("Error: could not load image in sift n' shift")

		target_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		 # Initialize the SIFT detector
		sift = cv2.SIFT_create()

		# Detect keypoints and descriptors in both images
		template_keypoints, template_descriptors = sift.detectAndCompute(self.template_image, None)
		image_keypoints, image_descriptors = sift.detectAndCompute(target_image, None)

		if template_descriptors is not None and image_descriptors is not None:
			# Use FLANN-based matcher for feature matching
			index_params = dict(algorithm=0, trees=5)
			search_params = dict()
			flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
			
			matches = flann.knnMatch(template_descriptors, image_descriptors, k=2)

			# Apply Lowe's ratio test to filter good matches
			good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

			print(len(good_matches))
			if len(good_matches) > 4:  # Homography needs at least 4 points
				src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) 
				dst_pts = np.float32([image_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

				# Find the homography matrix
				H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

				if H is not None:
					print("Found Homography")
					# Get the height and width of the template image
					h, w = self.template_image.shape

					# Define the corners of the template image
					template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

					# Use the homography to project the corners of the template onto the video frame
					projected_corners = cv2.perspectiveTransform(template_corners, H)

					# Draw the projected corners on the frame
					matched_frame = cv2.polylines(frame.copy(), [np.int32(projected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

				# Draw matches on the frame
				matched_frame = cv2.drawMatches(self.template_image, template_keypoints, frame, image_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
					
				perspective_fit_clue_board = cv2.warpPerspective(frame, np.linalg.inv(H), (w+400, h+220)) #400, 200 for clueboard1

				return perspective_fit_clue_board[50:-50, 100:-90] #sexy hardcoded values!

		# If no good matches or descriptors are found, return None
		return None


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())