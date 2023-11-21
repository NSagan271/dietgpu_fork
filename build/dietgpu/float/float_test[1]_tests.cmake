add_test( FloatTest.Batch /home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/bin/float_test [==[--gtest_filter=FloatTest.Batch]==] --gtest_also_run_disabled_tests)
set_tests_properties( FloatTest.Batch PROPERTIES WORKING_DIRECTORY /home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/dietgpu/float SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set( float_test_TESTS FloatTest.Batch)
