#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, CameraInfo
import ros_numpy

import numpy as np
import pcl
from scipy import stats
from sklearn.decomposition import PCA

import tf
import tf2_ros
import geometry_msgs.msg


def to_unit(vec):
    norm = np.linalg.norm(vec)
    return vec/norm


def xy_to_quaternion(x, y):
    """
    use perpendicular vectors x,y to defined rotation quaternion of the plane
    :param x: np.array.shape = (3,)
    :param y: np.array.shape = (3,)
    :return: np.array.shape = (4,)
    """
    z = np.cross(x, y)
    x = to_unit(x)
    y = to_unit(y)
    z = to_unit(z)
    rotmat = np.vstack((x, y, z)).transpose()
    rotmat_h = np.identity(4)
    rotmat_h[0:3, 0:3] = rotmat
    quat = tf.transformations.quaternion_from_matrix(rotmat_h)
    return quat


class Projector:
    def __init__(self, model):
        """model[a,b,c,d] plane: ax+by+cz+d=0"""
        n = model[0:3]
        norm = np.linalg.norm(n)
        dist = -model[3]/norm
        normal_vec = n/norm
        center = normal_vec*dist
        plane_y = np.cross(normal_vec, np.array([1, 0, 0]))
        plane_y /= np.linalg.norm(plane_y)
        plane_x = np.cross(plane_y, normal_vec)
        plane_x /= np.linalg.norm(plane_x)
        plane_to_world = np.identity(4)
        plane_to_world[0:3, 0] = plane_x
        plane_to_world[0:3, 1] = plane_y
        plane_to_world[0:3, 2] = normal_vec
        plane_to_world[0:3, 3] = center

        world_to_plane = np.linalg.inv(plane_to_world)
        self.plane_to_world = plane_to_world
        self.world_to_plane = world_to_plane

    def to_plane(self, cloud_arr):
        """
        transform 3d points to 2d coordinate on plane
        :param cloud_arr: np.array().shape = (3, n)
        :return: np.array().shape = (2,n)
        """
        if cloud_arr.shape[0] != 3:
            raise RuntimeError('Incorrect cloud_arr shape')
        world_to_plane = self.world_to_plane
        cloud_h = np.vstack((cloud_arr, np.ones((1, cloud_arr.shape[1]))))
        cloud_plane = np.matmul(world_to_plane, cloud_h)
        plane_arr = cloud_plane[0:3]/cloud_plane[3]
        plane_arr = plane_arr[0:2]
        return plane_arr

    def to_world(self, mat):
        """
        transform 2d points on plane back to world frame
        :param mat: np.array().shape = (2,n)
        :return: np.array().shape = (3, n)
        """
        if mat.shape[0] != 2:
            raise RuntimeError('Incorrect cloud_arr shape')
        plane_to_world = self.plane_to_world
        mat_h = np.vstack((mat, np.zeros((1, mat.shape[1])), np.ones((1, mat.shape[1]))))
        mat_world_h = np.matmul(plane_to_world, mat_h)
        mat_world = mat_world_h[0:3]/mat_world_h[3]
        return mat_world

    def to_world_rot_only(self, mat):
        """
        transform vectors on 2d coordinate back to world frame
        :param mat: np.array().shape = (2,n)
        :return: np.array().shape = (3, n)
        """
        if mat.shape[0] != 2:
            raise RuntimeError('Incorrect cloud_arr shape')
        plane_to_world = self.plane_to_world
        rot_mat = plane_to_world[0:3, 0:3]
        mat = np.vstack((mat, np.zeros((1, mat.shape[1]))))
        mat_world = np.matmul(rot_mat, mat)
        return mat_world


def ransac_plane(orig_cloud, distance_threshold = 0.007):
    """
    find a plane in point cloud
    :param orig_cloud: PCL::PointCloud
    :param distance_threshold: float
    :return: [int], [a,b,c,d](ax+by+cz+d=0)
    """
    seg = orig_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    indices, model = seg.segment()
    return indices, model


def component_min_max(xyvec, plane):
    projected = np.matmul(xyvec.transpose(), plane)
    # trimmed = stats.trimboth(projected, 0.01)
    trimmed = projected
    print(trimmed.min(), trimmed.max())
    return trimmed.min(), trimmed.max()


def stats_outlier_remove(cloud):
    """
    filter points 3stdev away
    :param cloud: PCL::PointCloud
    :return: PCL::PointCloud
    """
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(3)
    return fil.filter()


def find_candidate_plane(pcl_cloud, table_ratio=1.428, ratio_threshold = 0.02):
    """
    Iteratively find a plane containing table
    :param pcl_cloud: PCL::PointCloud
    :param table_ratio: expected table ratio (width/height)
    :param ratio_threshold:
    :return: PCL::PointCloud, [a,b,c,d](ax+by+cz+d=0)
    """
    on_table = None
    table_model = None

    pca = PCA(n_components=3)
    i = 0
    while True:
        if i > 30:
            raise RuntimeError('Failed to find candidate table plane.')
        i += 1

        # find plane
        indices, model = ransac_plane(pcl_cloud)
        inliers = pcl_cloud.extract(indices, negative=False)

        # remove points 3 stdev away
        inliers_filtered = stats_outlier_remove(inliers)
        inliers_filtered_arr = inliers_filtered.to_array()

        # check if plane is table using ratio of two sides
        pca.fit(inliers_filtered_arr)
        sv = pca.singular_values_
        sv_ratio = sv[0]/sv[1]
        print('candidate ratio:', sv_ratio)
        is_table = abs(sv_ratio-table_ratio) < ratio_threshold

        if is_table:
            # found table
            on_table = inliers
            table_model = model
            break
        else:
            # not table
            pcl_cloud = pcl_cloud.extract(indices, negative=True)

    return on_table, table_model


def pca_localization(plane):
    """
    :param plane: plane projected to 2 by n ndarray
    :return: ndarray([[xvec][yvec][center]).transpose()
    """
    pca = PCA(n_components=2)
    pca.fit(plane.transpose())
    comp = pca.components_
    xvec = None
    yvec = None
    plane_y = np.array([0, 1])
    if abs(np.dot(comp[0], plane_y)) > abs(np.dot(comp[1], plane_y)):
        # comp[0] is vertical
        yvec = comp[0]
        xvec = comp[1]
    else:
        xvec = comp[0]
        yvec = comp[1]

    xmin, xmax = component_min_max(xvec, plane)
    ymin, ymax = component_min_max(yvec, plane)
    xcenter = (xmin + xmax) / 2.0
    ycenter = (ymin + ymax) / 2.0
    center = xvec*xcenter + yvec*ycenter
    lowerleft = xvec*xmin + yvec*ymax

    # plt.scatter(orig_plane[0], orig_plane[1])
    plt.scatter(plane[0], plane[1])
    plt.arrow(center[0], center[1], xvec[0], xvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01,
              color='r')
    plt.arrow(center[0], center[1], yvec[0], yvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01,
              color='r')
    angle = np.arctan2(xvec[1], xvec[0]) * 180 / np.pi
    rect = plt.Rectangle(lowerleft, (xmax - xmin), (ymax - ymin), fill=False, color='r', angle=angle)
    ca = plt.gca()
    ca.add_patch(rect)
    plt.show()
    out = np.vstack((xvec, yvec)).transpose()
    return out, center.reshape((1, 2)).transpose()


def tf2_broadcast(trans, quat, parent_frame_id='kinect2_victor_head_ir_optical_frame', child_frame_id='table'):
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent_frame_id
    static_transformStamped.child_frame_id = child_frame_id
    # trans = [1.00619416, -0.19541004, -0.42314096]
    # trans = [-0.01299984, -0.13024425,  0.95459]
    # trans = [-0.00621743, -0.13153817,  0.9555135]
    static_transformStamped.transform.translation.x = trans[0]
    static_transformStamped.transform.translation.y = trans[1]
    static_transformStamped.transform.translation.z = trans[2]
    # quat = tf.transformations.quaternion_from_euler(0, 0, 0)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]

    broadcaster.sendTransform(static_transformStamped)
    rospy.spin()


# ===================capture pointcloud2=======================
def msg_to_pcl(msg):
    arr = ros_numpy.numpify(msg)
    arr1d = arr.reshape(arr.shape[0]*arr.shape[1])
    reca = arr1d.view(np.recarray)
    newarr = np.vstack((reca.x,reca.y,reca.z))
    finalarr = np.transpose(newarr)
    cloud = pcl.PointCloud()
    cloud.from_array(finalarr)
    # pdb.set_trace()
    return cloud
# ===================end capture pointcloud2=======================


# ===================visualization========================
import matplotlib.pyplot as plt


def draw(plt, plane, center, xvec, yvec):
    plt.scatter(plane[0], plane[1])
    plt.arrow(center[0], center[1], xvec[0], xvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01, color='r')
    plt.arrow(center[0], center[1], yvec[0], yvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01, color='r')
    plt.show()
# ================end visualization=======================


def main(pcl_cloud):
    # find a plane that looks like a table
    on_plane, model = find_candidate_plane(pcl_cloud)
    on_plane_filtered = stats_outlier_remove(on_plane)
    on_plane_arr = on_plane.to_array().T
    on_plane_filtered_arr = on_plane_filtered.to_array().T

    # build transformation between 2d(on plane) and 3d
    proj = Projector(model)
    on_plane_filtered_arr_2d = proj.to_plane(on_plane_filtered_arr)

    # find 2d pose and center of the table using PCA
    xy2d, center2d = pca_localization(on_plane_filtered_arr_2d)

    # transform into 3d world frame
    xy3d = proj.to_world_rot_only(xy2d)
    center3d = proj.to_world(center2d)
    transp = xy3d.transpose()
    x = transp[0]
    y = transp[1]
    center = center3d.transpose()[0]
    quat = xy_to_quaternion(x, y)

    # start broadcast static transformation
    tf2_broadcast(center, quat, parent_frame_id='kinect2_victor_head_ir_optical_frame', child_frame_id='table')

GLOBAL_MSG_RECEIVED = False

def callback(msg):
    # print(msg)
    global GLOBAL_MSG_RECEIVED
    if not GLOBAL_MSG_RECEIVED:
        GLOBAL_MSG_RECEIVED = True
        cloud = msg_to_pcl(msg)
        main(cloud)


def listener():
    rospy.init_node('table_calibration_tf2_broadcaster', anonymous=False)
    rospy.Subscriber('/kinect2_victor_head/sd/points', PointCloud2, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def test(fname):
    pcl_cloud = pcl.load(fname)
    main(pcl_cloud)


listener()
