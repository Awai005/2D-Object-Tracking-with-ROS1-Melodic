find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  tf
)

find_package(Eigen3 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(object_tracker_cpp src/object_tracker.cpp)
target_link_libraries(object_tracker_cpp ${catkin_LIBRARIES})
add_dependencies(object_tracker_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
