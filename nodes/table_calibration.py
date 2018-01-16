#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy

import numpy as np
import pcl
from scipy import stats
from sklearn.decomposition import PCA

import tf
import tf2_ros
import geometry_msgs.msg

import pdb

def msg_to_pcl(msg):
    arr = ros_numpy.numpify(msg)
    arr1d = arr.reshape(arr.shape[0]*arr.shape[1])
    reca = arr1d.view(np.recarray)
    newarr = np.vstack((reca.x,reca.y,reca.z,reca.rgb))
    finalarr = np.transpose(newarr)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_array(finalarr)
    pdb.set_trace()
    return cloud

def ransac_plane(cloud, distance_threshold = 0.007):
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    indices, model = seg.segment()
    # print(model)
    # print(indices)
    pdb.set_trace()
    on_plane = cloud.extract(indices)
    return on_plane, model

def callback(msg):
    # print(msg)
    cloud = msg_to_pcl(msg)
    on_plane, model = ransac_plane(cloud)
    

def listener():

    rospy.init_node('pcl_listener', anonymous=True)

    rospy.Subscriber('kinect2_victor_head/sd/points', PointCloud2, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
