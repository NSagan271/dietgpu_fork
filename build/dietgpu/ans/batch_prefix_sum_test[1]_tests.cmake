add_test( BatchPrefixSum.OneLevel /home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/bin/batch_prefix_sum_test [==[--gtest_filter=BatchPrefixSum.OneLevel]==] --gtest_also_run_disabled_tests)
set_tests_properties( BatchPrefixSum.OneLevel PROPERTIES WORKING_DIRECTORY /home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/dietgpu/ans SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
add_test( BatchPrefixSum.TwoLevel /home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/bin/batch_prefix_sum_test [==[--gtest_filter=BatchPrefixSum.TwoLevel]==] --gtest_also_run_disabled_tests)
set_tests_properties( BatchPrefixSum.TwoLevel PROPERTIES WORKING_DIRECTORY /home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/dietgpu/ans SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set( batch_prefix_sum_test_TESTS BatchPrefixSum.OneLevel BatchPrefixSum.TwoLevel)
