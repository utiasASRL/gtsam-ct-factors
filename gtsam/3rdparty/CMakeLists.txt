# install CCOLAMD and SuiteSparse_config headers
install(FILES CCOLAMD/Include/ccolamd.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/3rdparty/CCOLAMD)
install(FILES SuiteSparse_config/SuiteSparse_config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/3rdparty/SuiteSparse_config)

# install Eigen unless System Eigen is used
if(NOT GTSAM_USE_SYSTEM_EIGEN)
	# Find plain .h files
	file(GLOB_RECURSE eigen_headers "${CMAKE_CURRENT_SOURCE_DIR}/Eigen/Eigen/*.h")

	# Find header files without extension
	file(GLOB eigen_dir_headers_all "Eigen/Eigen/*")
	foreach(eigen_dir ${eigen_dir_headers_all})
		get_filename_component(filename ${eigen_dir} NAME)
		if (NOT ((${filename} MATCHES "CMakeLists.txt") OR (${filename} MATCHES "src") OR (${filename} MATCHES ".svn")))
			list(APPEND eigen_headers "${CMAKE_CURRENT_SOURCE_DIR}/Eigen/Eigen/${filename}")
			install(FILES Eigen/Eigen/${filename} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/3rdparty/Eigen/Eigen)
		endif()
	endforeach(eigen_dir)

	if(GTSAM_WITH_EIGEN_UNSUPPORTED)
		message(STATUS "Installing Eigen Unsupported modules")
		# do the same for the unsupported eigen folder
		file(GLOB_RECURSE  unsupported_eigen_headers "${CMAKE_CURRENT_SOURCE_DIR}/Eigen/unsupported/Eigen/*.h")

		file(GLOB unsupported_eigen_dir_headers_all "Eigen/unsupported/Eigen/*")
		foreach(unsupported_eigen_dir ${unsupported_eigen_dir_headers_all})
			get_filename_component(filename ${unsupported_eigen_dir} NAME)
			if (NOT ((${filename} MATCHES "CMakeLists.txt") OR (${filename} MATCHES "src") OR (${filename} MATCHES "CXX11") OR (${filename} MATCHES ".svn")))
				list(APPEND unsupported_eigen_headers "${CMAKE_CURRENT_SOURCE_DIR}/Eigen/unsupported/Eigen/${filename}")
				install(FILES Eigen/unsupported/Eigen/${filename} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/3rdparty/Eigen/unsupported/Eigen)
			endif()
		endforeach(unsupported_eigen_dir)
	endif()

	# Add to project source
	set(eigen_headers ${eigen_headers} PARENT_SCOPE)
	# set(unsupported_eigen_headers ${unsupported_eigen_headers} PARENT_SCOPE)

	# install Eigen - only the headers in our 3rdparty directory
	install(DIRECTORY Eigen/Eigen
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/3rdparty/Eigen
		FILES_MATCHING PATTERN "*.h")

	if(GTSAM_WITH_EIGEN_UNSUPPORTED)
		install(DIRECTORY Eigen/unsupported/Eigen
			DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtsam/3rdparty/Eigen/unsupported/
			FILES_MATCHING PATTERN "*.h")
	endif()

endif()

# metis: already handled in ROOT/cmake/HandleMetis.cmake

add_subdirectory(ceres)

############ NOTE:  When updating GeographicLib be sure to disable building their examples
############        and unit tests by commenting out their lines:
# add_subdirectory (examples)
# set (TOOLS CartConvert ConicProj GeodesicProj GeoConvert GeodSolve
#   GeoidEval Gravity MagneticField Planimeter TransverseMercatorProj)
# add_subdirectory (tools)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Try to find an already-installed GeographicLib.
find_package(GeographicLib QUIET)

if(GeographicLib_FOUND AND NOT TARGET GeographicLib::GeographicLib)
  add_library(GeographicLib::GeographicLib UNKNOWN IMPORTED)
  set_target_properties(GeographicLib::GeographicLib PROPERTIES
    IMPORTED_LOCATION             "${GeographicLib_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${GeographicLib_INCLUDE_DIRS}"
  )
endif()

if(TARGET GeographicLib::GeographicLib)
  message(STATUS "Using system GeographicLib: ${GeographicLib_LIBRARIES}")
else()
  message(STATUS "GeographicLib not found; GeographicLib-dependent tests will be disabled.")
endif()
