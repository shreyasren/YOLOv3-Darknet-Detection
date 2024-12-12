#include "YoloClassification.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <chrono>
#include <iostream>
#include <atomic>
#include <thread>
#include <fstream>
#include <mutex>


using namespace cv;

namespace yolo_classification
{

  const char *COCO_CLASSES_[] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

  YoloClassification::YoloClassification(ros::NodeHandle n, ros::NodeHandle pn)
  {
    pn.param("darknet_cfg_file", darknet_cfg_file_, std::string(""));
    pn.param("darknet_weights_file", darknet_weights_file_, std::string(""));

    srv_.setCallback(boost::bind(&YoloClassification::reconfig, this, _1, _2));
    net_ = load_network(const_cast<char*>(darknet_cfg_file_.c_str()), const_cast<char*>(darknet_weights_file_.c_str()), 0);

    sub_image_ = n.subscribe("image_rect", 1, &YoloClassification::recvImage, this);
    pub_detections_ = n.advertise<YoloObjectArray>("yolo_objects", 1);

    skip_ = 0;
  }

  void YoloClassification::recvImage(const sensor_msgs::ImageConstPtr& msg)
  {
    // Skip frames to save computational bandwidth
    if (skip_ >= cfg_.skip) {
      skip_ = 0;
    } else {
      skip_++;
      return;
    }

    // Convert ROS image message into an OpenCV Mat
    Mat raw_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;

    // Run image through the neural network and return classifications and bounding boxes
    YoloObjectArray darknet_bboxes;
    darknet_bboxes.header = msg->header;
    runDarknet(raw_img, darknet_bboxes.objects);

    // Publish detections for other nodes to use
    pub_detections_.publish(darknet_bboxes);
  }

  // std::ofstream non_tbb_data_file("/home/shreyas/ros/src/YOLOv3-Darknet-Detection/data/runDarknet_runtime_vs_threads.csv");
  // std::mutex non_tbb_data_mutex;
  // bool non_tbb_header_written = false;

  // void YoloClassification::runDarknet(const Mat& raw_img, std::vector<YoloObject>& darknet_bboxes)
  // {
  //     auto start = std::chrono::high_resolution_clock::now();

  //     ROS_INFO_STREAM("Thread ID: " << std::this_thread::get_id());

  //     image temp_img = mat_to_image(raw_img);
  //     image im = resize_image(temp_img, net_->w, net_->h);
  //     free_image(temp_img);

  //     double image_scale_x = (double)raw_img.cols / (double)im.h;
  //     double image_scale_y = (double)raw_img.rows / (double)im.w;
  //     set_batch_network(net_, 1);
  //     network_predict(net_, im.data);
  //     int nboxes;
  //     detection* dets = get_network_boxes(net_, im.w, im.h, cfg_.thres, cfg_.hier, NULL, 0, &nboxes);
  //     free_image(im);

  //     for (int i = 0; i < nboxes; i++) {
  //         int best_classification = -1;
  //         double highest_prob = -INFINITY;
  //         for (int j = 0; j < dets[i].classes; j++) {
  //             double prob = dets[i].prob[j];
  //             if ((prob > cfg_.min_prob) && (prob > highest_prob)) {
  //                 highest_prob = prob;
  //                 best_classification = j;
  //             }
  //         }

  //         if (best_classification < 0) {
  //             continue;
  //         }

  //         box b = dets[i].bbox;
  //         int left = (int)(image_scale_x * (b.x - 0.5 * b.w));
  //         int right = (int)(image_scale_x * (b.x + 0.5 * b.w));
  //         int top = (int)(image_scale_y * (b.y - 0.5 * b.h));
  //         int bot = (int)(image_scale_y * (b.y + 0.5 * b.h));

  //         YoloObject candidate_bbox;
  //         candidate_bbox.label = std::string(COCO_CLASSES_[best_classification]);
  //         candidate_bbox.confidence = highest_prob;
  //         candidate_bbox.x = left;
  //         candidate_bbox.y = top;
  //         candidate_bbox.w = right - left;
  //         candidate_bbox.h = bot - top;

  //         bool found_duplicate = false;
  //         for (auto& bbox : darknet_bboxes) {
  //             int dx = std::abs(bbox.x - left);
  //             int dy = std::abs(bbox.y - top);
  //             if ((dx < cfg_.duplicate_thres) && (dy < cfg_.duplicate_thres)) {
  //                 found_duplicate = true;
  //                 break;
  //             }
  //         }

  //         if (!found_duplicate) {
  //             darknet_bboxes.push_back(candidate_bbox);
  //         }
  //     }
  //     free_detections(dets, nboxes);

  //     auto end = std::chrono::high_resolution_clock::now();
  //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  //     ROS_INFO_STREAM("Standard Implementation runtime: " << duration << " ms");

  //     // Write execution time to CSV file
  //     {
  //         std::lock_guard<std::mutex> lock(non_tbb_data_mutex);

  //         if (!non_tbb_header_written) {
  //             non_tbb_data_file << "Threads,Runtime(ms)\n";
  //             non_tbb_header_written = true;
  //         }

  //         // Since this is a single-threaded function, Threads = 1
  //         non_tbb_data_file << "1," << duration << "\n";
  //     }
  // }
  

  // Global variables for thread-safe CSV writing
  std::ofstream data_file("/home/shreyas/ros/src/YOLOv3-Darknet-Detection/data/runDarknet_runtime_vs_threads.csv");
  std::mutex data_mutex;

  void YoloClassification::runDarknet(const Mat& raw_img, std::vector<YoloObject>& darknet_bboxes) 
  {
      auto start = std::chrono::high_resolution_clock::now();

      image temp_img = mat_to_image(raw_img);
      image im = resize_image(temp_img, net_->w, net_->h);
      free_image(temp_img);

      double image_scale_x = (double)raw_img.cols / (double)im.h;
      double image_scale_y = (double)raw_img.rows / (double)im.w;
      set_batch_network(net_, 1);
      network_predict(net_, im.data);

      int nboxes;
      detection* dets = get_network_boxes(net_, im.w, im.h, cfg_.thres, cfg_.hier, NULL, 0, &nboxes);
      free_image(im);

      std::atomic<int> thread_counter{0};

      tbb::parallel_for(tbb::blocked_range<int>(0, nboxes), [&](const tbb::blocked_range<int>& range) {
          for (int i = range.begin(); i < range.end(); ++i) {
              int thread_index = tbb::this_task_arena::current_thread_index();
              if (thread_index >= 0) {
                  thread_counter.fetch_add(1, std::memory_order_relaxed);
              }

              // Perform object classification and bounding box extraction
              int best_classification = -1;
              double highest_prob = -INFINITY;
              for (int j = 0; j < dets[i].classes; j++) {
                  double prob = dets[i].prob[j];
                  if ((prob > cfg_.min_prob) && (prob > highest_prob)) {
                      highest_prob = prob;
                      best_classification = j;
                  }
              }

              if (best_classification < 0) continue;

              box b = dets[i].bbox;
              int left  = (int)(image_scale_x * (b.x - 0.5 * b.w));
              int right = (int)(image_scale_x * (b.x + 0.5 * b.w));
              int top   = (int)(image_scale_y * (b.y - 0.5 * b.h));
              int bot   = (int)(image_scale_y * (b.y + 0.5 * b.h));

              YoloObject candidate_bbox;
              candidate_bbox.label = std::string(COCO_CLASSES_[best_classification]);
              candidate_bbox.confidence = highest_prob;
              candidate_bbox.x = left;
              candidate_bbox.y = top;
              candidate_bbox.w = right - left;
              candidate_bbox.h = bot - top;

              darknet_bboxes.push_back(candidate_bbox);
          }
      });

      free_detections(dets, nboxes);

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      ROS_INFO_STREAM("Parallel Implementation runtime: " << duration << " ms");
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

  void YoloClassification::reconfig(DarknetConfig& config, uint32_t level) {
    cfg_ = config;
  }

}