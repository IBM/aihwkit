#include "../src/rpucuda/rng.h"
#include "../src/rpucuda/rpu_constantstep_device.h"
#include "../src/rpucuda/utility_functions.h"
#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <vector>

namespace {

using namespace RPU;

class HSInspectableConstantStepDevice : public ConstantStepRPUDevice<num_t> {
public:
  using ConstantStepRPUDevice<num_t>::ConstantStepRPUDevice;
  using PulsedRPUDeviceBase<num_t>::allocateHSContainers;
  using PulsedRPUDeviceBase<num_t>::freeHSContainers;
  using PulsedRPUDeviceBase<num_t>::resetHSStates;
  using PulsedRPUDeviceBase<num_t>::updateHSStateOnly;
};

class HSStateTrackingFixture : public ::testing::Test {
protected:
  void SetUp() override {
    dp_.dw_min = 0.1;
    dp_.dw_min_std = 0.0;
    dp_.dw_min_dtod = 0.0;
    dp_.up_down_dtod = 0.0;
    dp_.w_max = 10.0;
    dp_.w_min = -10.0;
    dp_.w_max_dtod = 0.0;
    dp_.w_min_dtod = 0.0;
    dp_.hs_decay = 0.5;

    device_ = std::make_unique<HSInspectableConstantStepDevice>(x_size_, d_size_, dp_, &rw_rng_);
    weights_ = Array_2D_Get<num_t>(d_size_, x_size_);
    for (int i = 0; i < d_size_; i++) {
      for (int j = 0; j < x_size_; j++) {
        weights_[i][j] = 1.0;
      }
    }
    ASSERT_TRUE(device_->onSetWeights(weights_));
    device_->enableHSTracking();
  }

  void TearDown() override {
    device_->disableHSTracking();
    Array_2D_Free(weights_);
  }

  static constexpr int x_size_ = 4;
  static constexpr int d_size_ = 4;
  ConstantStepRPUDeviceMetaParameter<num_t> dp_;
  RealWorldRNG<num_t> rw_rng_{11};
  RNG<num_t> rng_{7};
  std::unique_ptr<HSInspectableConstantStepDevice> device_;
  num_t **weights_ = nullptr;
};

struct HSTransitionCase {
  HalfSelectedState prev;
  HalfSelectedState curr;
  bool expected_decay;
  const char *name;
};

class HalfSelectedShouldDecayTest : public ::testing::TestWithParam<HSTransitionCase> {};

TEST_P(HalfSelectedShouldDecayTest, HalfSelectedShouldDecayRules) {
  ConstantStepRPUDeviceMetaParameter<num_t> dp;
  RealWorldRNG<num_t> rw_rng(3);
  HSInspectableConstantStepDevice device(1, 1, dp, &rw_rng);

  const auto &tc = GetParam();
  EXPECT_EQ(device.shouldApplyHSDecay(tc.prev, tc.curr), tc.expected_decay);
}

INSTANTIATE_TEST_CASE_P(
    HalfSelectedTransitions,
    HalfSelectedShouldDecayTest,
    ::testing::Values(
        HSTransitionCase{HalfSelectedState::HS1, HalfSelectedState::HS3, true, "HS1ToHS3"},
        HSTransitionCase{HalfSelectedState::HS3, HalfSelectedState::HS1, true, "HS3ToHS1"},
        HSTransitionCase{HalfSelectedState::HS2, HalfSelectedState::HS4, true, "HS2ToHS4"},
        HSTransitionCase{HalfSelectedState::HS4, HalfSelectedState::HS2, true, "HS4ToHS2"},
        HSTransitionCase{HalfSelectedState::HS1, HalfSelectedState::HS2, true, "HS1ToHS2"},
        HSTransitionCase{HalfSelectedState::HS2, HalfSelectedState::HS1, true, "HS2ToHS1"},
        HSTransitionCase{HalfSelectedState::HS3, HalfSelectedState::HS4, true, "HS3ToHS4"},
        HSTransitionCase{HalfSelectedState::HS4, HalfSelectedState::HS3, true, "HS4ToHS3"},
        HSTransitionCase{HalfSelectedState::HS1, HalfSelectedState::HS1, false, "HS1ToHS1"},
        HSTransitionCase{HalfSelectedState::HS0, HalfSelectedState::HS1, false, "HS0ToHS1"}),
    [](const ::testing::TestParamInfo<HSTransitionCase> &info) { return info.param.name; });

TEST_F(HSStateTrackingFixture, HalfSelectedMemoryLifecycle) {
  device_->disableHSTracking();
  EXPECT_FALSE(device_->isHSTrackingEnabled());
  EXPECT_EQ(device_->getHSStates(), nullptr);
  EXPECT_EQ(device_->getHSTransitionCounts(), nullptr);

  device_->allocateHSContainers();
  ASSERT_NE(device_->getHSStates(), nullptr);
  ASSERT_NE(device_->getHSTransitionCounts(), nullptr);

  device_->getHSStates()[0][0] = HalfSelectedState::HS4;
  device_->getHSTransitionCounts()[0][0] = 9;
  device_->resetHSStates();
  EXPECT_EQ(device_->getHSStates()[0][0], HalfSelectedState::HS1);
  EXPECT_EQ(device_->getHSTransitionCounts()[0][0], 0);

  device_->freeHSContainers();
  EXPECT_EQ(device_->getHSStates(), nullptr);
  EXPECT_EQ(device_->getHSTransitionCounts(), nullptr);

  device_->enableHSTracking();
  EXPECT_TRUE(device_->isHSTrackingEnabled());
  device_->disableHSTracking();
  EXPECT_FALSE(device_->isHSTrackingEnabled());
  EXPECT_EQ(device_->getHSStates(), nullptr);
  EXPECT_EQ(device_->getHSTransitionCounts(), nullptr);

  device_->enableHSTracking();
}

TEST_F(HSStateTrackingFixture, HalfSelectedStateClassification) {
  const int x_pos_col0[] = {1};
  const int x_neg_col1[] = {-2};

  device_->updateHSStateOnly(0, x_pos_col0, 1, 1, true, false);
  EXPECT_EQ(device_->getHSStates()[0][0], HalfSelectedState::HS1);

  device_->updateHSStateOnly(0, x_neg_col1, 1, 1, true, false);
  EXPECT_EQ(device_->getHSStates()[0][1], HalfSelectedState::HS3);

  device_->updateHSStateOnly(1, nullptr, 0, 1, false, true);
  for (int j = 0; j < x_size_; j++) {
    EXPECT_EQ(device_->getHSStates()[1][j], HalfSelectedState::HS2);
  }

  device_->updateHSStateOnly(2, nullptr, 0, -1, false, true);
  for (int j = 0; j < x_size_; j++) {
    EXPECT_EQ(device_->getHSStates()[2][j], HalfSelectedState::HS4);
  }
}

TEST_F(HSStateTrackingFixture, HalfSelectedDecayAppliedInSparseUpdateHS) {
  const int x_pos_col0[] = {1};
  const int x_pos_col1[] = {2};

  weights_[0][0] = 1.0;
  device_->getHSStates()[0][0] = HalfSelectedState::HS1;
  device_->doSparseUpdateHS(weights_, 0, x_pos_col0, 1, -1, &rng_);
  EXPECT_EQ(device_->getHSStates()[0][0], HalfSelectedState::HS3);
  EXPECT_NEAR(weights_[0][0], 0.6, 1e-6);

  weights_[0][1] = 1.0;
  device_->getHSStates()[0][1] = HalfSelectedState::HS0;
  device_->doSparseUpdateHS(weights_, 0, x_pos_col1, 1, 1, &rng_);
  EXPECT_EQ(device_->getHSStates()[0][1], HalfSelectedState::HS1);
  EXPECT_NEAR(weights_[0][1], 0.9, 1e-6);
}

TEST_F(HSStateTrackingFixture, HalfSelectedStateTrackingOn4x4For10BLSteps) {
  PulsedUpdateMetaParameter<num_t> up;
  up.pulse_type = PulseType::HalfselectedStochastic;
  EXPECT_EQ(up.pulse_type, PulseType::HalfselectedStochastic);

  const int x_col0_pos[] = {1};
  const int x_col1_neg[] = {-2};
  const int x_col2_pos[] = {3};
  const int x_col3_neg[] = {-4};

  int bl_steps = 0;

  device_->updateHSStateOnly(0, x_col0_pos, 1, 1, true, false);
  bl_steps++;
  device_->updateHSStateOnly(0, x_col1_neg, 1, 1, true, false);
  bl_steps++;
  device_->updateHSStateOnly(1, nullptr, 0, 1, false, true);
  bl_steps++;
  device_->updateHSStateOnly(2, nullptr, 0, -1, false, true);
  bl_steps++;
  device_->updateHSStateOnly(3, x_col2_pos, 1, 1, true, false);
  bl_steps++;
  device_->updateHSStateOnly(3, x_col3_neg, 1, 1, true, false);
  bl_steps++;

  device_->doSparseUpdateHS(weights_, 0, x_col0_pos, 1, -1, &rng_);
  bl_steps++;
  device_->doSparseUpdateHS(weights_, 0, x_col1_neg, 1, -1, &rng_);
  bl_steps++;

  device_->updateHSStateOnly(0, nullptr, 0, 1, false, true);
  bl_steps++;
  device_->updateHSStateOnly(0, nullptr, 0, -1, false, true);
  bl_steps++;

  ASSERT_EQ(bl_steps, 10);

  for (int j = 0; j < x_size_; j++) {
    EXPECT_EQ(device_->getHSStates()[0][j], HalfSelectedState::HS4);
    EXPECT_EQ(device_->getHSStates()[1][j], HalfSelectedState::HS2);
    EXPECT_EQ(device_->getHSStates()[2][j], HalfSelectedState::HS4);
  }
  EXPECT_EQ(device_->getHSStates()[3][0], HalfSelectedState::HS1);
  EXPECT_EQ(device_->getHSStates()[3][1], HalfSelectedState::HS1);
  EXPECT_EQ(device_->getHSStates()[3][2], HalfSelectedState::HS1);
  EXPECT_EQ(device_->getHSStates()[3][3], HalfSelectedState::HS3);

  std::vector<int> transition_counts;
  device_->getHSTransitionCounts(transition_counts);
  ASSERT_EQ(transition_counts.size(), 16);
  int total_transitions = 0;
  for (int c : transition_counts) {
    total_transitions += c;
  }
  EXPECT_GT(total_transitions, 0);
  EXPECT_NEAR(weights_[0][0], 0.6, 1e-6);
  EXPECT_NEAR(weights_[0][1], 0.4, 1e-6);
}

}
