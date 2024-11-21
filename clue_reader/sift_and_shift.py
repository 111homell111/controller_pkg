import cv2
import numpy as np
import rospkg
import rospy

class Sift_and_Shifter():
	def __init__(self):
		
		#import the clue_template
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('controller_pkg') 
		template_image_path = f"{package_path}/clue_reader/assets/clue_template.png" 
		self.template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
		if self.template_image is None:
			rospy.logerr("Failed to load template image.")
		else:
			rospy.loginfo(f"Loaded template image from {template_image_path}")
		


	def set_template_image(self, template_image):
		self.template_image = template_image

	def get_template_image(self):
		return self.template_image.copy()

	def sift_and_shift(self, raw_frame):
		"""
		@brief Takes the fizz clue rule symbol and performs sift to get a homography. Then does an inverse perspective transform and crops 
				the image to isolate the clue board.
				Uncomment matched frame if you want to see the SIFT feature lines

		@param frame: raw camera footage in OpenCV format
		@return perspective transformed and isolated/cropped image of the clue board
		"""

		if self.template_image is None or raw_frame is None:
			print("Error: could not load image in sift n' shift")

		target_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

		sift = cv2.SIFT_create()

		# Detect keypoints and descriptors in both images
		template_keypoints, template_descriptors = sift.detectAndCompute(self.template_image, None)
		image_keypoints, image_descriptors = sift.detectAndCompute(target_image, None)

		if template_descriptors is not None and image_descriptors is not None:
			index_params = dict(algorithm=0, trees=5)
			search_params = dict()
			flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
			
			matches = flann.knnMatch(template_descriptors, image_descriptors, k=2)

			# Apply Lowe's ratio test to filter good matches
			good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

			if len(good_matches) > 4:  # Homography needs at least 4 points
				src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) 
				dst_pts = np.float32([image_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

				H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

				if H is not None:
					print("Found Homography")
					h, w = self.template_image.shape
					template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
					# Use the homography to project the corners of the template onto the video frame
					projected_corners = cv2.perspectiveTransform(template_corners, H)
					#matched_frame = cv2.polylines(raw_frame.copy(), [np.int32(projected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

				#matched_frame = cv2.drawMatches(self.template_image, template_keypoints, raw_frame, image_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
					

				#THESE ARE MY SEXY HARDCODED CROPPING VALUES!
				perspective_fit_clue_board = cv2.warpPerspective(raw_frame, np.linalg.inv(H), (w+400, h+220)) #400, 200 for clueboard1
				return perspective_fit_clue_board[50:-50, 100:-90]

		# If no good matches or descriptors are found, return None
		return None
