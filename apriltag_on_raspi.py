#!/usr/bin/env python3

# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.

import json
import time
import sys
import cv2
from dt_apriltags import Detector
import numpy as np
import math

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from ntcore import NetworkTableInstance, EventFlags

configFile = "/boot/frc.json"

class CameraConfig: pass

team = 2052
server = False
cameraConfigs = []
cameras = []

# Server config and stuff

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readCameraConfig(config):
    """Read single camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get("stream")

    cam.config = config

    cameraConfigs.append(cam)
    return True

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

def startCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    camera = UsbCamera(config.name, config.path)
    server = CameraServer.startAutomaticCapture(camera=camera)

    camera.setConfigJson(json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kConnectionKeepOpen)

    if config.streamConfig is not None:
        server.setConfigJson(json.dumps(config.streamConfig))

    return camera

####################################################################################

class Tag():

    def __init__(self, tag_size, family):
        self.family = family
        self.size = tag_size
        self.locations = {}
        self.orientations = {}
        self.found_tags0 = []
        self.found_tags1 = []
        corr = np.eye(3)
        corr[0, 0] = -1
        self.tag_corr = corr
        
    def addTag(self,id,x,y,z,theta_x,theta_y,theta_z):
        self.locations[id]=self.inchesToTranslationVector(x,y,z)
        self.orientations[id]=self.eulerAnglesToRotationMatrix(theta_x,theta_y,theta_z)
        
    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta_x,theta_y,theta_z):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]
                        ])

        R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]
                        ])

        R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]
                        ])

        R = np.matmul(R_z, np.matmul(R_y, R_x))

        return np.transpose(R)
        
    def inchesToTranslationVector(self,x,y,z):
        #inches to meters
        return np.array([[x],[y],[z]])*0.0254
    
    def addFoundTags(self, tags, cam):
        if cam == 0:
            self.found_tags0 = []
            for tag in tags:
                if tag.hamming < 9:
                    self.found_tags0.append(tag)
        if cam == 1:
            self.found_tags1 = []
            for tag in tags:
                if tag.hamming < 9:
                    self.found_tags1.append(tag)

    def getFilteredTags(self, cam):
        if cam == 0:
            return self.found_tags0
        if cam == 1:
            return self.found_tags1

    def estimateTagPose(self, tag_id, R,t):
        local = self.tag_corr @ np.transpose(R) @ t
        return np.matmul(self.orientations[tag_id], local + self.locations[tag_id])

####################################################################################

# Visualize input frame with detected tags

def visualize_frame(img, tags):
    color_img = img

    for tag in tags:
        # Add bounding rectangle
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                    (0, 255, 0), thickness=3)
        # Add Tag ID text
        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),
                    thickness=3)
        # Add Tag Corner
        cv2.circle(color_img, tuple(tag.corners[0].astype(int)), 2, color=(255, 0, 255), thickness=3)
    
    return color_img

####################################################################################

# Pose Estimation

class PoseEstimator():

    def estimatePoseMeters(self):
        
        estimated_poses_list0 = []
        estimated_poses_list1 = []

        for tag in tags.getFilteredTags(0):
            estimated_poses_list0.append(tags.estimateTagPose(tag.tag_id, tag.pose_R, tag.pose_t))
        for tag in tags.getFilteredTags(1):
            estimated_poses_list1.append(tags.estimateTagPose(tag.tag_id, tag.pose_R, tag.pose_t))
        
        if not estimated_poses_list0 or not estimated_poses_list1:
            print("no estimated poses list")
            # If we have no samples, report none
            return (None, estimated_poses_list0)
                
        total0 = np.array([0.0, 0.0, 0.0])
        total1 = np.array([0.0, 0.0, 0.0])

        for pose in estimated_poses_list0:
            total0 = ([(total[0] + pose[0]), (total[1] + pose[1]), (total[2] + pose[2])])
        average0 = np.divide(total, len(estimated_poses_list0))

        for pose in estimated_poses_list1:
            total1 = ([(total[0] + pose[0]), (total[1] + pose[1]), (total[2] + pose[2])])
        average1 = np.divide(total, len(estimated_poses_list1))

        return ([average0, average1])
        
####################################################################################

# Constants

# Camera Parameters:
camera_info = {}
# [fx, fy, cx, cy] f is focal length in pixels x and y, c is optical center in pixels x and y.
# focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * PI/180)

# 1280,960 res: **UNTESTED**
#camera_info["params"] = [1338.26691, 1338.26691, 639.266524, 486.552512] 
#RES = (1280,960)
# 640,480 res: 
camera_info["params"] = [669.13345619, 669.13345619, 319.63326201, 243.27625621]
RES = (640,480)
# 320,240 res: 
#camera_info["params"] = [334.566728095, 334.566728095, 159.816631005, 121.638128105]
#RES = (320,240)

camera_info["res"] = RES

inchesInAMeter = 39.37

TAG_SIZE = (6.5)/39.37
FAMILIES = "tag36h11"

tags = Tag(TAG_SIZE, FAMILIES)
poseEstimator = PoseEstimator()

####################################################################################

# INITIALIZE VARIABLES #

# add tags:
# takes in id,x,y,z,theta_x,theta_y,theta_z
# theta_x:roll, theta_y:pitch, theta_z:yaw

tags.addTag(0,0,0,0,0,0,0)
tags.addTag(1, 0., 17.5, 0., 0., 0., 0)
tags.addTag(2, 0., 18.22, 0., 0., 0., 180)
tags.addTag(3, 0., 18.22, 0., 0., 0., 180)
tags.addTag(4, 0., 27.38, 0., 0., 0., 180)
tags.addTag(5, 0., 27.38, 0., 0., 0., 0.)
tags.addTag(6, 0., 18.22, 0., 0., 0., 0.)
tags.addTag(7, 0., 18.22, 0., 0., 0., 0.)
tags.addTag(8, 0., 18.22, 0., 0., 0., 0.)

####################################################################################

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTableInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient4("wpilibpi")
        ntinst.setServerTeam(team)
        ntinst.startDSClient()

    raspberryPiTable = ntinst.getTable("RaspberryPi")

    # start cameras
    # work around wpilibsuite/allwpilib#5055
    CameraServer.setSize(CameraServer.kSize640x480)
    for config in cameraConfigs:
        cameras.append(startCamera(config))

    detector0 = Detector(families='tag36h11',
                        nthreads=4,
                        quad_decimate=2,
                        quad_sigma=0,
                        refine_edges=1,
                        decode_sharpening=0,
                        debug=0
                        )
    detector1 = Detector(families='tag36h11',
                        nthreads=4,
                        quad_decimate=2,
                        quad_sigma=0,
                        refine_edges=1,
                        decode_sharpening=0,
                        debug=0
                        )
    

    cvSink0 = CameraServer.getVideo("rPi Camera 0")
    cvSink1 = CameraServer.getVideo("rPi Camera 1")

    processedOutput0 = CameraServer.putVideo("processedOutput0", RES[0], RES[1])
    processedOutput1 = CameraServer.putVideo("processedOutput1", RES[0], RES[1])

    frame0 = np.zeros(shape=(RES[1], RES[0], 3), dtype=np.uint8)
    frame1 = np.zeros(shape=(RES[1], RES[0], 3), dtype=np.uint8)


    # loop forever
    while True:
        frameTime, frame0 = cvSink0.grabFrame(frame0)
        if frameTime == 0:
            processedOutput1.notifyError(cvSink1.getError())

            continue
        frameTime, frame1 = cvSink1.grabFrame(frame1)
        if frameTime == 0:
            processedOutput2.notifyError(cvSink2.getError())

            continue

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        detected_tags0 = detector0.detect(gray0, estimate_tag_pose=True, camera_params=camera_info["params"], tag_size=TAG_SIZE)
        detected_tags1 = detector1.detect(gray1, estimate_tag_pose=True, camera_params=camera_info["params"], tag_size=TAG_SIZE)

        #t1 = time()
        tags.addFoundTags(detected_tags0, 0)
        tags.addFoundTags(detected_tags1, 1)
        
        processedOutput0.putFrame(visualize_frame(frame0, tags.getFilteredTags(0)))
        processedOutput1.putFrame(visualize_frame(frame1, tags.getFilteredTags(1)))
        
        """
        outputTags1 = []
        outputTags2 = []
        for tag in tags.getFilteredTags():
            ID = tag.tag_id
            R = tag.pose_R
            t = tag.pose_t

            yaw = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])))
            pitch =  math.degrees(math.atan2(R[1,2]/math.cos(yaw), R[2,2]/math.cos(yaw)))
            roll = math.degrees(math.atan2(R[1,0]/math.cos(yaw), R[0,0]/math.cos(yaw)))

            # convert meters to inches (inches * 39.37)
            x = (int((t[0]).astype(float)*inchesInAMeter*1000)) / 1000
            y = (int((t[1]).astype(float)*inchesInAMeter*1000)) / 1000
            z = (int((t[2]).astype(float)*inchesInAMeter*1000)) / 1000
        """
            
        # pose detector gives x (left and right), y (up and down),z (forward backward)
        # field relative: x (points away away from driverstation aka forward backward) y (perpendicular to x aka left and right)
        pose = poseEstimator.estimatePoseMeters()
        if pose[0] != None:
            raspberryPiTable.putNumberArray("camera0PoseMeters", [pose[0][2], pose[0][0], pose[0][1]])
            raspberryPiTable.putNumberArray("camera1PoseMeters", [pose[1][2], pose[1][0], pose[1][1]])
        
####################################################################################

"""

DETECTOR PARAMETERS:

PARAMETERS:
families : Tag families, separated with a space, default: tag36h11

nthreads : Number of threads, default: 1

quad_decimate : Detection of quads can be done on a lower-resolution image, 
improving speed at a cost of pose accuracy and a slight decrease in detection rate. 
Decoding the binary payload is still done at full resolution, default: 2.0

quad_sigma : What Gaussian blur should be applied to the segmented image (used for 
quad detection?) Parameter is the standard deviation in pixels. Very noisy images 
benefit from non-zero values (e.g. 0.8), default: 0.0

refine_edges : When non-zero, the edges of the each quad are adjusted to “snap to” 
strong gradients nearby. This is useful when decimation is employed, as it can increase 
the quality of the initial quad estimate substantially. Generally recommended to be 
on (1). Very computationally inexpensive. Option is ignored if quad_decimate = 0, 
default: 1

decode_sharpening : How much sharpening should be done to decoded images? This can 
help decode small tags but may or may not help in odd lighting conditions or low light 
conditions, default = 0.25

debug : If 1, will save debug images. Runs very slow, default: 0
"""
