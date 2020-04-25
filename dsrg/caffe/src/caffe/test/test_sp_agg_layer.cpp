#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sp_agg_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <iostream>
namespace caffe {
/* Superpixel aggregation layer, compute mean feature in a superpixel
    bottom[0]: fc8 (N, 21, 41, 41)
    bottom[1]: sp (N, 1, 41, 41)
    bottom[2]: num_sp (N, 1, 1, 1)
    top[0]: sp_agg_fc8 (1, 1, 21, N_sp_all)
*/
template <typename TypeParam>
class SpAggLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SpAggLayerTest()
      : blob_bottom_fc8(new Blob<Dtype>()),
        blob_bottom_sp(new Blob<Dtype>()),
        blob_bottom_num_sp(new Blob<Dtype>()),
        blob_top_sp_agg(new Blob<Dtype>()) {

    Caffe::set_random_seed(1701);
    blob_bottom_fc8->Reshape(2, 2, 3, 3);
    // fill the values
    // FillerParameter filler_param;
    // GaussianFiller<Dtype> filler(filler_param);
    // filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_fc8);
    blob_bottom_vec_.push_back(blob_bottom_sp);
    blob_bottom_vec_.push_back(blob_bottom_num_sp);
    blob_top_vec_.push_back(blob_top_sp_agg);
  }
  virtual ~SpAggLayerTest() {
    delete blob_bottom_fc8;
    delete blob_bottom_sp;
    delete blob_bottom_num_sp;
    delete blob_top_sp_agg;
  }
  Blob<Dtype>* const blob_bottom_fc8;
  Blob<Dtype>* const blob_bottom_sp;
  Blob<Dtype>* const blob_bottom_num_sp;
  Blob<Dtype>* const blob_top_sp_agg;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    int num = 2;
    int channels = 2;
    int height = 3;
    int width = 3;
    LayerParameter layer_param;
    blob_bottom_fc8->Reshape(num, channels, height, width);
    blob_bottom_sp->Reshape(num, 1, height, width);
    blob_bottom_num_sp->Reshape(num, 1, 1, 1);
    blob_top_sp_agg->Reshape(1, 1, channels, 7);
    /*  num 1 data
        7  4  3        9  0  4
        2  5  1.5      2  3  7
        8  6.4 4       1  5  0.5
        num 1 mask
        0  1  1
        2  1  3
        2  3  3

        num 2 data
        5  2  4       7  8  0.2
        0.3 1 0.5     0.3 5 1
        2  6  4.3     4  3  2
        num 2 mask
        0  0  0
        0  1  1
        2  2  2
    */    
    blob_bottom_fc8->mutable_cpu_data()[0] = 7;
    blob_bottom_fc8->mutable_cpu_data()[1] = 4;
    blob_bottom_fc8->mutable_cpu_data()[2] = 3;
    blob_bottom_fc8->mutable_cpu_data()[3] = 2;
    blob_bottom_fc8->mutable_cpu_data()[4] = 5;
    blob_bottom_fc8->mutable_cpu_data()[5] = 1.5f;
    blob_bottom_fc8->mutable_cpu_data()[6] = 8;
    blob_bottom_fc8->mutable_cpu_data()[7] = 6.4f;
    blob_bottom_fc8->mutable_cpu_data()[8] = 4;
    blob_bottom_fc8->mutable_cpu_data()[9] = 9;
    blob_bottom_fc8->mutable_cpu_data()[10] = 0;
    blob_bottom_fc8->mutable_cpu_data()[11] = 4;
    blob_bottom_fc8->mutable_cpu_data()[12] = 2;
    blob_bottom_fc8->mutable_cpu_data()[13] = 3;
    blob_bottom_fc8->mutable_cpu_data()[14] = 7;
    blob_bottom_fc8->mutable_cpu_data()[15] = 1;
    blob_bottom_fc8->mutable_cpu_data()[16] = 5;
    blob_bottom_fc8->mutable_cpu_data()[17] = 0.5f;
    blob_bottom_fc8->mutable_cpu_data()[18] = 5;
    blob_bottom_fc8->mutable_cpu_data()[19] = 2;
    blob_bottom_fc8->mutable_cpu_data()[20] = 4;
    blob_bottom_fc8->mutable_cpu_data()[21] = 0.3f;
    blob_bottom_fc8->mutable_cpu_data()[22] = 1;
    blob_bottom_fc8->mutable_cpu_data()[23] = 0.5f;
    blob_bottom_fc8->mutable_cpu_data()[24] = 2;
    blob_bottom_fc8->mutable_cpu_data()[25] = 6;
    blob_bottom_fc8->mutable_cpu_data()[26] = 4.3f;
    blob_bottom_fc8->mutable_cpu_data()[27] = 7;
    blob_bottom_fc8->mutable_cpu_data()[28] = 8;
    blob_bottom_fc8->mutable_cpu_data()[29] = 0.2f;
    blob_bottom_fc8->mutable_cpu_data()[30] = 0.3f;
    blob_bottom_fc8->mutable_cpu_data()[31] = 5;
    blob_bottom_fc8->mutable_cpu_data()[32] = 1;
    blob_bottom_fc8->mutable_cpu_data()[33] = 4;
    blob_bottom_fc8->mutable_cpu_data()[34] = 3;
    blob_bottom_fc8->mutable_cpu_data()[35] = 2;
    
    blob_bottom_sp->mutable_cpu_data()[0] = 0;
    blob_bottom_sp->mutable_cpu_data()[1] = 1;
    blob_bottom_sp->mutable_cpu_data()[2] = 1;
    blob_bottom_sp->mutable_cpu_data()[3] = 2;
    blob_bottom_sp->mutable_cpu_data()[4] = 1;
    blob_bottom_sp->mutable_cpu_data()[5] = 3;
    blob_bottom_sp->mutable_cpu_data()[6] = 2;
    blob_bottom_sp->mutable_cpu_data()[7] = 3;
    blob_bottom_sp->mutable_cpu_data()[8] = 3;
    blob_bottom_sp->mutable_cpu_data()[9] = 0;
    blob_bottom_sp->mutable_cpu_data()[10] = 0;
    blob_bottom_sp->mutable_cpu_data()[11] = 0;
    blob_bottom_sp->mutable_cpu_data()[12] = 0;
    blob_bottom_sp->mutable_cpu_data()[13] = 1;
    blob_bottom_sp->mutable_cpu_data()[14] = 1;
    blob_bottom_sp->mutable_cpu_data()[15] = 2;
    blob_bottom_sp->mutable_cpu_data()[16] = 2;
    blob_bottom_sp->mutable_cpu_data()[17] = 2;

    blob_bottom_num_sp->mutable_cpu_data()[0] = 4;
    blob_bottom_num_sp->mutable_cpu_data()[1] = 3;

    SpAggLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_sp_agg->num(), 1);
    EXPECT_EQ(blob_top_sp_agg->channels(), 1);
    EXPECT_EQ(blob_top_sp_agg->height(), channels);
    EXPECT_EQ(blob_top_sp_agg->width(), 7);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    Dtype epsilon = 1e-7;
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[0], 7, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[1], 4, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[2], 5, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[3], Dtype(11.9)/3, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[4], Dtype(11.3)/4, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[5], Dtype(1.5)/2, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[6], Dtype(4.1), epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[7], 9, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[8], Dtype(7.0)/3, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[9], Dtype(3.0)/2, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[10], Dtype(12.5)/3, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[11], Dtype(15.5)/4, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[12], 3, epsilon);
    EXPECT_NEAR(blob_top_sp_agg->cpu_data()[13], 3, epsilon);
    
  }

  void TestBackward()
  {
    int num = 2;
    int channels = 2;
    int height = 3;
    int width = 3;
    LayerParameter layer_param;
    blob_bottom_fc8->Reshape(num, channels, height, width);
    blob_bottom_sp->Reshape(num, 1, height, width);
    blob_bottom_num_sp->Reshape(num, 1, 1, 1);
    blob_top_sp_agg->Reshape(1, 1, channels, 7);
    std::cout << "test_sp" << 181 << std::endl;
    for(int h = 0; h < num; h++)
    {
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 0] = 1;
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 1] = 3;
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 2] = 2;
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 3] = 3;
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 4] = 4;
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 5] = 2;
      blob_top_sp_agg->mutable_cpu_diff()[h*7 + 6] = 3;
    }
    std::cout << "test_sp" << 192 << std::endl;
    blob_bottom_sp->mutable_cpu_data()[0] = 0;
    blob_bottom_sp->mutable_cpu_data()[1] = 1;
    blob_bottom_sp->mutable_cpu_data()[2] = 1;
    blob_bottom_sp->mutable_cpu_data()[3] = 2;
    blob_bottom_sp->mutable_cpu_data()[4] = 1;
    blob_bottom_sp->mutable_cpu_data()[5] = 3;
    blob_bottom_sp->mutable_cpu_data()[6] = 2;
    blob_bottom_sp->mutable_cpu_data()[7] = 3;
    blob_bottom_sp->mutable_cpu_data()[8] = 3;
    blob_bottom_sp->mutable_cpu_data()[9] = 0;
    blob_bottom_sp->mutable_cpu_data()[10] = 0;
    blob_bottom_sp->mutable_cpu_data()[11] = 0;
    blob_bottom_sp->mutable_cpu_data()[12] = 0;
    blob_bottom_sp->mutable_cpu_data()[13] = 1;
    blob_bottom_sp->mutable_cpu_data()[14] = 1;
    blob_bottom_sp->mutable_cpu_data()[15] = 2;
    blob_bottom_sp->mutable_cpu_data()[16] = 2;
    blob_bottom_sp->mutable_cpu_data()[17] = 2;

    blob_bottom_num_sp->mutable_cpu_data()[0] = 4;
    blob_bottom_num_sp->mutable_cpu_data()[1] = 3;

    SpAggLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    vector<bool> propagate_down;
    propagate_down.push_back(true);
    propagate_down.push_back(false);
    propagate_down.push_back(false);
    layer.Reshape(blob_bottom_vec_, blob_top_vec_);
    layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);
    Dtype epsilon = 1e-7;
    for(int i = 0; i < num;  ++i)
    {
      for(int c = 0; c < channels; ++c)
      {
        for(int h = 0; h < height; ++h)
        {
          for(int w = 0; w < width; ++w)
          {
            //std::cout<<"num = " << i << " cha==" << c << " h=" << h << " w=" << w << std::endl;
            EXPECT_NEAR(blob_bottom_fc8->cpu_diff()[blob_bottom_fc8->offset(i,c,h,w)], 1, epsilon);
          }
        }
      }
    }
    
  }

};

 TYPED_TEST_CASE(SpAggLayerTest, TestDtypesAndDevices);

 TYPED_TEST(SpAggLayerTest, TestCPUForward) {
   //typedef typename TypeParam::Dtype Dtype;
   //  Caffe::set_mode(Caffe::CPU);
   this->TestForward();
 }

 TYPED_TEST(SpAggLayerTest, TestCPUBackward) {
   //typedef typename TypeParam::Dtype Dtype;
   //  Caffe::set_mode(Caffe::CPU);
   this->TestBackward();
}

} // namespace caffe