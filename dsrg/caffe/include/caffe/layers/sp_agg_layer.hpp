#ifndef CAFFE_SP_AGG_LAYER
#define CAFFE_SP_AGG_LAYER

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {
/* Superpixel aggregation layer, compute mean feature in a superpixel
    bottom[0]: fc8 (N, 21, 41, 41)
    bottom[1]: sp (N, 1, 41, 41)
    bottom[2]: num_sp (N, 1, 1, 1)
    top[0]: sp_agg_fc8 (1, 1, 21, N_sp_all)
*/
template <typename Dtype>
class SpAggLayer : public Layer<Dtype> {
 public:
  explicit SpAggLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "SpAgg"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int height_, width_, channels_, num_;
  vector<int> pixelcount_; // num of pixels in each superpixel
  vector<int> sp_start_idx_; // start index of first superpixel for each image
  int num_sp_all_;

}; // class SpAggLayer
}// namespace caffe


#endif
