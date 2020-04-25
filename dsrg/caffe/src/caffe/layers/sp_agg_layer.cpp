#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sp_agg_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
namespace caffe {
/* Superpixel aggregation layer, compute mean feature in a superpixel
    bottom[0]: fc (N, C, H, W)
    bottom[1]: sp (N, 1, H, W)
    bottom[2]: num_sp (N, 1, 1, 1)
    top[0]: sp_agg_fc (1, 1, C, N_sp_all)
*/
template <typename Dtype>
void SpAggLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  this->channels_ = bottom[0]->channels();
  this->num_ = bottom[0]->num();
}

template <typename Dtype>
void SpAggLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // compute num_sp_all_
  const Dtype* num_sp = bottom[2]->cpu_data();
  this->num_sp_all_ = 0;
  for(int i = 0; i < this->num_; ++i)
  {
    this->num_sp_all_ += (int)num_sp[i];
  }
  // First Net set up Reshape call
  if(this->num_sp_all_ == 0)
  {
    this->num_sp_all_ = 100;
  }
  top[0]->Reshape(1, 1, this->channels_, this->num_sp_all_);

  // compute sp_start_idx
  this->sp_start_idx_.resize(this->num_);
  this->sp_start_idx_[0] = 0;
  for(int i = 1; i < this->num_; ++i)
  {
    this->sp_start_idx_[i] = this->sp_start_idx_[i-1] + (int)num_sp[i-1];
  }
  // compute pixelcount
  this->pixelcount_.resize(this->num_sp_all_);
  for(int s = 0; s < this->num_sp_all_; ++s)
  {
    this->pixelcount_[s] = 0;
  }
  for(int i = 0; i < this->num_; ++i)
  {
    for(int h = 0; h < this->height_; ++h)
    {
      for(int w = 0; w < this->width_; ++w)
      {
        int sp_idx = this->sp_start_idx_[i] + 
            (int)bottom[1]->data_at(i, 0, h, w);
        this->pixelcount_[sp_idx] += 1;
      }
    }
  }
}
/*
    bottom[0]: fc (N, C, H, W)
    bottom[1]: sp (N, 1, H, W)
    bottom[2]: num_sp (N, 1, 1, 1)
    top[0]: sp_agg_fc (1, 1, C, N_sp_all)
*/
template <typename Dtype>
void SpAggLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //int num_sp_all = top[0]->width();
  const Dtype* fc_data = bottom[0]->cpu_data();
  Dtype* sp_agg_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.0), sp_agg_data);
  //const Dtype* sp_seg = bottom[1]->cpu_data();
  for(int i = 0; i < this->num_; ++i)
  {
    for(int h = 0; h < this->height_; ++h)
    {
      for(int w = 0; w < this->width_; ++w)
      {
        int sp_idx = this->sp_start_idx_[i] + 
             (int)bottom[1]->data_at(i, 0, h, w);
        // this->pixelcount_[sp_idx] += 1;
        for(int c = 0; c < this->channels_; ++c)
        {
          sp_agg_data[top[0]->offset(0, 0, c, sp_idx)] += 
            fc_data[bottom[0]->offset(i,c,h,w)];
        }
      }
    }
  }

  for(int s = 0;  s < this->num_sp_all_; ++s)
  {
    for(int c = 0; c < this->channels_; ++c)
    {
      sp_agg_data[top[0]->offset(0,0,c,s)] /= this->pixelcount_[s];
    }
  }
}

/*
    bottom[0]: fc8 (N, C, H, W)
    bottom[1]: sp (N, 1, H, W)
    bottom[2]: num_sp (N, 1, 1, 1)
    top[0]: sp_agg_fc8 (1, 1, C, N_sp_all)
*/
template <typename Dtype>
void SpAggLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0])
  {
    for(int i = 0; i < this->num_; ++i)
    {
      for(int h = 0; h < this->height_; ++h)
      {
        for(int w = 0; w < this->width_; ++w)
        {
          int sp_idx = this->sp_start_idx_[i] + 
              (int)bottom[1]->data_at(i, 0, h, w);
          for(int c = 0; c < this->channels_; ++c)
          {
            bottom_diff[bottom[0]->offset(i, c, h, w)] = 
              top_diff[top[0]->offset(0, 0, c, sp_idx)] / this->pixelcount_[sp_idx];
          }
        }
      }
    }
  }

} 

INSTANTIATE_CLASS(SpAggLayer);
REGISTER_LAYER_CLASS(SpAgg);
}// namespace caffe