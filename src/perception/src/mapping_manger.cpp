#include "mapping_manager.h"

namespace mot_mapping
{

    MappingRos::MappingRos(ros::NodeHandle &nh) : nh(nh) {}
    MappingRos::~MappingRos() { std::cout << "Exit Tracker" << std::endl; }

    void MappingRos::init()
    {
      
      t_start = ros::Time::now();
      id = 0;
      mp_ = std::make_unique<MapParam>();
      md_ = std::make_unique<MapData>();

      objectPosePub = nh.advertise<visualization_msgs::MarkerArray>("/object_pose", 10);
      statesPub = nh.advertise<obj_state_msgs::ObjectsStates>("/states", 10);

      cloud_sub_.reset(
          new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/vision_detector/target_position", 50));
      odom_sub_.reset(
          new message_filters::Subscriber<nav_msgs::Odometry>(nh, "/Odometry", 25));
      sync_cloud_odom_.reset(new message_filters::Synchronizer<MappingRos::SyncPolicyCloudOdom>(
          MappingRos::SyncPolicyCloudOdom(100), *cloud_sub_, *odom_sub_));
      sync_cloud_odom_->registerCallback(boost::bind(&MappingRos::cloudOdomCallback, this, _1, _2));

      ekf_predict_timer_ = nh.createTimer(ros::Duration(0.02), &MappingRos::ekfPredictCallback, this);
  }


  void MappingRos::cloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &msg,
                                     const nav_msgs::OdometryConstPtr &odom)
  {
    // ====================== pose transformation ======================
    // double compTime;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    // body -> lidar_init
    Eigen::Vector3d t_body_in_lidar(
      odom->pose.pose.position.x,
      odom->pose.pose.position.y,
      odom->pose.pose.position.z
    );
    Eigen::Quaterniond q_body_in_lidar(
      odom->pose.pose.orientation.w,
      odom->pose.pose.orientation.x,
      odom->pose.pose.orientation.y,
      odom->pose.pose.orientation.z
    );
    q_body_in_lidar.normalize();
    Eigen::Matrix3d R_body_in_lidar = q_body_in_lidar.toRotationMatrix();

    // camera_init -> body
    Eigen::Vector3d t_cam_in_body(0.05, 0.0, -0.03);
    Eigen::Quaterniond q_cam_in_body(0.5, 0.5, 0.5, 0.5); // w, x, y, z
    q_cam_in_body.normalize(); 
    Eigen::Matrix3d R_cam_in_body = q_cam_in_body.toRotationMatrix();

    vision_targets.clear();
    vision_targets.reserve(cloud.size());
    for (const auto& pt : cloud.points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            continue;
        }
        Eigen::Vector3d p_cam(pt.x, pt.y, pt.z);
        // camera_init -> body
        Eigen::Vector3d p_body = R_cam_in_body * p_cam + t_cam_in_body;
        // body -> lidar_init
        Eigen::Vector3d p_lidar = R_body_in_lidar * p_body + t_body_in_lidar;
        vision_targets.push_back(p_lidar);
    }

    // =========================== Tracking object initialization ===========================
    if (trackers.size() == 0 && vision_targets.size() != 0)
    {
      for (int i = 0; i < vision_targets.size(); ++i)
      {
        std::shared_ptr<Ekf> ekfPtr = std::make_shared<Ekf>(0.02);
        ekfPtr->reset(vision_targets[i], id);
        trackers.push_back(ekfPtr);
        id++;
        previous_p.push_back(vision_targets[i]);
        Eigen::Vector3d v0(0,0,0);
        previous_v.push_back(v0);
      }
      return;
    }
    
    // =========================== Tracking object inter frame association ===========================
    std::vector<int> delete_tracker;
    std::vector<int> tra_vis;
    std::vector<bool> vision_matched(vision_targets.size(), false);
    if(trackers.size() != 0 && vision_targets.size() != 0){
      for (int i = 0; i < trackers.size(); ++i)
      {
        double min_dist = 9999;
        int min_idx = -1;
        for (int j = 0; j < vision_targets.size(); ++j)
        {
          double dist = (trackers[i]->pos() - vision_targets[j]).norm();
          if (min_dist > dist)
          {
            min_dist = dist;
            min_idx = j;
          }
        }
        if (min_dist > 0.05 && min_dist < 0.5 && min_idx != -1)
        {
          double dt = (ros::Time::now() - trackers[i]->last_update_stamp_).toSec();
          if (dt <= 1e-3 || dt > 0.5)
          {
            dt = 0.02;
          }
          
          vision_matched[min_idx] = true; 
          Eigen::Vector3d vel_detect = (vision_targets[min_idx] - previous_p[i]) / dt;
          trackers[i]->update(vision_targets[min_idx], 0.2*vel_detect + 0.8 * previous_v[i]);
          trackers[i]->age = trackers[i]->age + 1;
          tra_vis.push_back(trackers[i]->id);
        }
        else
        {
          delete_tracker.push_back(i);
        }
      }
      for (int i = 0; i < vision_targets.size(); ++i)
      {
        if(!vision_matched[i])
        {
          std::shared_ptr<Ekf> ekfPtr = std::make_shared<Ekf>(0.02);
          ekfPtr->reset(vision_targets[i], id);
          trackers.push_back(ekfPtr);
          tra_vis.push_back(id);
          id++;
        }
      }
    }
    std::sort(delete_tracker.begin(), delete_tracker.end(), std::greater<int>());
    for (int idx : delete_tracker) {
        if (idx >= 0 && idx < trackers.size()) {
            trackers.erase(trackers.begin() + idx);
        }
    }

    previous_p.clear();
    previous_v.clear();
    for (int i = 0; i < trackers.size(); ++i ) {
      previous_p.push_back(trackers[i]->pos());
      previous_v.push_back(trackers[i]->vel());
    }
    visualizeFunction(tra_vis);
  }

  void MappingRos::ekfPredictCallback(const ros::TimerEvent &e)
  {
    if (trackers.size() == 0)
      return;
    for (int i = 0; i < trackers.size(); ++i)
    {
      trackers[i]->predict();
    }
  }

  void MappingRos::visualizeFunction(const std::vector<int> tra_id)
  {
    visualization_msgs::MarkerArray poses;
    obj_state_msgs::ObjectsStates states;
    for (const int& idx : tra_id) {
      for (int i = 0; i < trackers.size(); ++i)
      {
        if (idx == trackers[i]->id)
        {
          visualization_msgs::Marker poseMarker;
          ObjectState state;
          state.pos = trackers[i]->pos();
          state.vel = trackers[i]->vel();
          poseMarker.id = trackers[i]->id;
          poseMarker.header.stamp = ros::Time::now();
          poseMarker.header.frame_id = "lidar_init";
          poseMarker.lifetime = ros::Duration(0.1);
          poseMarker.type = visualization_msgs::Marker::ARROW;
          poseMarker.action = visualization_msgs::Marker::ADD;
          poseMarker.ns = "objectpose";
          poseMarker.color.r = 0.00;
          poseMarker.color.g = 1.00;
          poseMarker.color.b = 0.00;
          poseMarker.color.a = 1.00;
          poseMarker.scale.x = 0.10;
          poseMarker.scale.y = 0.18;
          poseMarker.scale.z = 0.30;
          poseMarker.pose.orientation.w = 1.0;
          geometry_msgs::Point arrow[2];
          arrow[0].x = state.pos[0];
          arrow[0].y = state.pos[1];
          arrow[0].z = state.pos[2];
          arrow[1].x = state.pos[0] + 1.0 * state.vel[0] / state.vel.norm();
          arrow[1].y = state.pos[1] + 1.0 * state.vel[1] / state.vel.norm();
          arrow[1].z = state.pos[2] + 1.0 * state.vel[2] / state.vel.norm();
          poseMarker.points.push_back(arrow[0]);
          poseMarker.points.push_back(arrow[1]);
          if (state.vel.norm() > 0.06)
            poses.markers.push_back(poseMarker);
          
          obj_state_msgs::State statemsg;
          statemsg.header.stamp = ros::Time::now();
          statemsg.position.x = state.pos[0]; statemsg.position.y = state.pos[1]; statemsg.position.z = state.pos[2];
          statemsg.velocity.x = state.vel[0]; statemsg.velocity.y = state.vel[1]; statemsg.velocity.z = state.vel[2];
          statemsg.size.x = 0; statemsg.size.y = 0; statemsg.size.z = 0;
          states.states.push_back(statemsg);
        }
      }
    }
    objectPosePub.publish(poses);
    statesPub.publish(states);
  }
}
