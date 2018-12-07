##
# Build file for Darknet include directory
##
set( headers
  yolo_v2_class.hpp
  )

install( FILES ${headers}
  DESTINATION "include/darknet"
  )
