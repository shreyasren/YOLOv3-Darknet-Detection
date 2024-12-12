#include "SyncedYoloData.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tbb/parallel_for_each.h>
#include <atomic>
#include <tbb/task_arena.h>
#include <thread>
#include <fstream>
#include <mutex>

using namespace cv;

// Global atomic counter to count active threads
std::atomic<int> active_threads{0};

namespace yolo_classification
{

  SyncedYoloData::SyncedYoloData(ros::NodeHandle& n, ros::NodeHandle& pn)
  {
    sub_img_.reset(new message_filters::Subscriber<sensor_msgs::Image>(n, "image_rect", 5));
    sub_objects_.reset(new message_filters::Subscriber<YoloObjectArray>(n, "yolo_objects", 5));
    sync_yolo_data_.reset(new message_filters::Synchronizer<YoloSyncPolicy>(YoloSyncPolicy(10), *sub_img_, *sub_objects_));
    sync_yolo_data_->registerCallback(boost::bind(&SyncedYoloData::recvSyncedData, this, _1, _2));

    pub_output_image_ = n.advertise<sensor_msgs::Image>("yolo_image", 1);
  }

    // std::ofstream synced_data_file("/home/shreyas/ros/src/YOLOv3-Darknet-Detection/data/recvSyncedData_runtime_vs_threads.csv");
    // std::mutex synced_data_mutex;
    // bool synced_header_written = false;

    // void SyncedYoloData::recvSyncedData(const sensor_msgs::ImageConstPtr& img_msg, const YoloObjectArrayConstPtr& object_msg)
    // {
    //     auto start = std::chrono::high_resolution_clock::now();

    //     // Log the thread ID (should be single-threaded)
    //     ROS_INFO_STREAM("Thread ID: " << std::this_thread::get_id());

    //     Mat raw_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;

    //     for (auto& bbox : object_msg->objects) {
    //         cv::Point2d corner(bbox.x, bbox.y - 5);
    //         rectangle(raw_img, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), Scalar(0, 255, 0), 3);
    //         putText(raw_img, bbox.label, corner, FONT_HERSHEY_DUPLEX, 0.75, Scalar(255, 255, 255));
    //     }

    //     sensor_msgs::ImagePtr output_img_msg = cv_bridge::CvImage(img_msg->header, "bgr8", raw_img).toImageMsg();
    //     pub_output_image_.publish(output_img_msg);

    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //     ROS_INFO_STREAM("Bounding box time for serial implementation: " << duration << " ms");

    //     // Write execution time to CSV file
    //     {
    //         std::lock_guard<std::mutex> lock(synced_data_mutex);

    //         if (!synced_header_written) {
    //             synced_data_file << "Threads,Runtime(ms)\n";
    //             synced_header_written = true;
    //         }
    //         synced_data_file << "1," << duration << "\n"; // Single-threaded, so Threads = 1
    //     }
    // }

  // Global variables for thread-safe CSV writing
  std::ofstream data_file("/home/shreyas/ros/src/YOLOv3-Darknet-Detection/data/recvSyncedData_runtime_vs_threads.csv");
  std::mutex data_mutex;

  void SyncedYoloData::recvSyncedData(const sensor_msgs::ImageConstPtr& img_msg, const YoloObjectArrayConstPtr& object_msg) 
  {
      auto start = std::chrono::high_resolution_clock::now();

      Mat raw_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;

      std::atomic<int> thread_counter{0};

      tbb::parallel_for_each(object_msg->objects.begin(), object_msg->objects.end(), [&](const auto& bbox) {
          // Increment active thread counter
          int thread_index = tbb::this_task_arena::current_thread_index();
          if (thread_index >= 0) {
              thread_counter.fetch_add(1, std::memory_order_relaxed);
          }

          // Perform bounding box drawing and labeling
          cv::Point2d corner(bbox.x, bbox.y - 5);
          rectangle(raw_img, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), Scalar(0, 255, 0), 3);
          putText(raw_img, bbox.label, corner, FONT_HERSHEY_DUPLEX, 0.75, Scalar(255, 255, 255));
      });

      // Publish the processed image
      sensor_msgs::ImagePtr output_img_msg = cv_bridge::CvImage(img_msg->header, "bgr8", raw_img).toImageMsg();
      pub_output_image_.publish(output_img_msg);

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      ROS_INFO_STREAM("Bounding box time for parallel implementation: " << duration << " ms");
      ROS_INFO_STREAM("Number of threads used: " << thread_counter.load());

      // Write runtime and thread count to the CSV file
      {
          std::lock_guard<std::mutex> lock(data_mutex);
          data_file << thread_counter.load() << "," << duration << "\n";
      }
  }

  // Ensure the file is closed properly at the end of the program
  void closeCSVFile() {
      data_file.close();
  }

}