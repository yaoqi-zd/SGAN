#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>

#if CV_VERSION_MAJOR == 3
#include <opencv2/imgcodecs/imgcodecs.hpp>
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV
#define CV_THRESH_OTSU cv::THRESH_OTSU
#endif
#endif  // USE_OPENCV

#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/util/im_transforms.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

const float prob_eps = 0.01;

#ifdef USE_OPENCV

void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < brightness_prob) {
    CHECK_GE(brightness_delta, 0) << "brightness_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -brightness_delta, brightness_delta, &delta);
    AdjustBrightness(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    in_img.convertTo(*out_img, -1, 1, delta);
  } else {
    *out_img = in_img;
  }
}

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < contrast_prob) {
    CHECK_GE(upper, lower) << "contrast upper must be >= lower.";
    CHECK_GE(lower, 0) << "contrast lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustContrast(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img) {
  if (fabs(delta - 1.f) > 1e-3) {
    in_img.convertTo(*out_img, -1, delta, 0);
  } else {
    *out_img = in_img;
  }
  // LOG(INFO) << "----------------------in adjustcontrast-----------------";
  // LOG(INFO) << in_img.type();
}

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < saturation_prob) {
    CHECK_GE(upper, lower) << "saturation upper must be >= lower.";
    CHECK_GE(lower, 0) << "saturation lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustSaturation(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta - 1.f) != 1e-3) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the saturation.
    channels[1].convertTo(channels[1], -1, delta, 0);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < hue_prob) {
    CHECK_GE(hue_delta, 0) << "hue_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -hue_delta, hue_delta, &delta);
    AdjustHue(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the hue.
    channels[0].convertTo(channels[0], -1, 1, delta);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < random_order_prob) {
    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);
    CHECK_EQ(channels.size(), 3);

    // Shuffle the channels.
    std::random_shuffle(channels.begin(), channels.end());
    cv::merge(channels, *out_img);
  } else {
    *out_img = in_img;
  }
}

cv::Mat ApplyDistort(const cv::Mat& in_img, const DistortionParameter& param) {
  cv::Mat out_img = in_img;
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  if (prob > 0.5) {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob(),
                     param.brightness_delta());

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob(),
                   param.contrast_lower(), param.contrast_upper());

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob(),
                     param.saturation_lower(), param.saturation_upper());

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  } else {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob(),
                     param.brightness_delta());

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob(),
                     param.saturation_lower(), param.saturation_upper());

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob(),
                   param.contrast_lower(), param.contrast_upper());

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  }

  return out_img;
}
#endif  // USE_OPENCV

}  // namespace caffe
