if(EXISTS "/home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/dietgpu/float/float_test[1]_tests.cmake")
  include("/home/ee274_mfguo_nsagan/mfguo/dietgpu_fork/build/dietgpu/float/float_test[1]_tests.cmake")
else()
  add_test(float_test_NOT_BUILT float_test_NOT_BUILT)
endif()
