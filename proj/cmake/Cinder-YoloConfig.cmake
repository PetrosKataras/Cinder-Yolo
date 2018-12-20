if( NOT Cinder-Yolo )
	# directory paths
	get_filename_component( DARKNET_PATH "${CMAKE_CURRENT_LIST_DIR}/../../src/darknet/" ABSOLUTE )
	get_filename_component( CI_DARKNET_SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../src/cinder/darknet" ABSOLUTE )
	get_filename_component( CI_DARKNET_INCLUDE_PATH "${CI_DARKNET_SOURCE_PATH}" ABSOLUTE )
	get_filename_component( CI_DARKNET_PUBLIC_INCLUDE_PATH "${CI_DARKNET_SOURCE_PATH}/../.." ABSOLUTE )
	# darknet options
	option( USE_GPU "Enable Darknet gpu support through Cuda" FALSE )
	option( USE_OPENCV "Enable OpenCV" FALSE )
	option( USE_CUDNN "Enable CudaNN" FALSE )
	# compile darknet
	add_subdirectory( ${DARKNET_PATH} ${CMAKE_CURRENT_BINARY_DIR}/darknet )
	if( USE_GPU )
		target_compile_definitions( darknet_lib INTERFACE "-DGPU" )
		# unfortunately the CMakeLists.txt file for darknet uses the old
		# cmake syntax with 'include_directories' instead of the more robust 'target_include_directories'
		# should change that at some point but since I m lazy grab includes manually now.
		# this is needed for enabling the GPU path and pulls in the Cuda interface
		get_target_property( DARKNET_INCLUDES darknet_lib INCLUDE_DIRECTORIES )
		message( STATUS " INCLUDES : " ${DARKNET_INCLUDES} )
		target_include_directories( darknet_lib INTERFACE ${DARKNET_INCLUDES} )
	endif()
	if( USE_OPENCV )
		target_compile_definitions( darknet_lib INTERFACE "-DOPENCV" )
	endif()
	if( USE_CUDNN )
		target_compile_definitions( darknet_lib INTERFACE "-DCUDNN" )
	endif()
	
	# Cinder-Yolo	
	set( CI_DARKNET_SOURCES ${CI_DARKNET_SOURCE_PATH}/CinderYolo.cpp )
	add_library( Cinder-Yolo ${CI_DARKNET_SOURCES} )

	target_include_directories( Cinder-Yolo PRIVATE ${CI_DARKNET_SOURCE_PATH}  )
	target_include_directories( Cinder-Yolo PUBLIC ${CI_DARKNET_PUBLIC_INCLUDE_PATH} ${DARKNET_INCLUDES} )

	if( NOT TARGET cinder )
		include( "${CINDER_PATH}/proj/cmake/configure.cmake" )
		find_package( cinder REQUIRED PATHS 
			"${CINDER_PATH}/${CINDER_LIB_DIRECTORY}"
			"$ENV{CINDER_PATH}/${CINDER_LIB_DIRECTORY}" 
		)
	endif()
	target_link_libraries( Cinder-Yolo PRIVATE cinder darknet_lib )
endif()
