cmake_minimum_required(VERSION 3.5)
project(stand_go2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#list(APPEND CMAKE_PREFIX_PATH "libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/dspoljaric/libtorch/share/cmake/Torch")

include_directories(include)
link_directories(src)
#link_directories(libtorch)


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(Torch REQUIRED)

if (Torch_FOUND)
  message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")
  message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
  include_directories(${TORCH_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
  message(FATAL_ERROR "Torch not found")
endif()



set (
  DEPENDENCY_LIST
  unitree_go
  rclcpp
  std_msgs
)

add_executable(stand_go2 src/stand_go2.cpp src/motor_crc.cpp)
add_executable(policy_general src/policy_general.cpp src/motor_crc.cpp)
add_executable(policy_general_hw src/policy_general_hw.cpp src/motor_crc.cpp)

add_executable(policy_pos20 src/policy_pos20.cpp src/motor_crc.cpp)
add_executable(policy_pos20_hw src/policy_pos20_hw.cpp src/motor_crc.cpp)
add_executable(policy_pos50_hw src/policy_pos50_hw.cpp src/motor_crc.cpp)

ament_target_dependencies(stand_go2 ${DEPENDENCY_LIST})
ament_target_dependencies(policy_general ${DEPENDENCY_LIST} )
ament_target_dependencies(policy_general_hw ${DEPENDENCY_LIST} )
ament_target_dependencies(policy_pos20 ${DEPENDENCY_LIST} )
ament_target_dependencies(policy_pos20_hw ${DEPENDENCY_LIST} )
ament_target_dependencies(policy_pos50_hw ${DEPENDENCY_LIST} )


target_link_libraries(stand_go2  ${TORCH_LIBRARIES})
target_link_libraries(policy_general ${TORCH_LIBRARIES})
target_link_libraries(policy_general_hw  ${TORCH_LIBRARIES})
target_link_libraries(policy_pos20  ${TORCH_LIBRARIES})
target_link_libraries(policy_pos20_hw  ${TORCH_LIBRARIES})
target_link_libraries(policy_pos50_hw  ${TORCH_LIBRARIES})


set_property(TARGET stand_go2 PROPERTY CXX_STANDARD 17)


install(TARGETS 
        stand_go2
        policy_general
        policy_general_hw
        policy_pos20
        policy_pos20_hw
        policy_pos50_hw
        ${PROJECT_NAME})

#set_target_properties(policy_go2 PROPERTIES
#        BUILD_WITH_INSTALL_RPATH TRUE
#        INSTALL_RPATH_USE_LINK_PATH TRUE
#        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/libtorch/lib"
#    )

# set_target_properties(stand_go2 PROPERTIES
#     BUILD_WITH_INSTALL_RPATH TRUE
#     INSTALL_RPATH_USE_LINK_PATH TRUE
#     INSTALL_RPATH "${CMAKE_SOURCE_DIR}/libtorch/lib"
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # uncomment the line when a copyright and license is not p
  # the following line skips the linter which checks for copyrightsresent in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()