#ifndef IM_TRANSFORMS_HPP
#define IM_TRANSFORMS_HPP

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#ifdef USE_OPENCV

void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta);

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper);

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img);

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper);

void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta);

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob);

cv::Mat ApplyDistort(const cv::Mat& in_img, const DistortionParameter& param);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // IM_TRANSFORMS_HPP
