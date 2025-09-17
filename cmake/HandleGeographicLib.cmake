unset(GeographicLib_INCLUDE_DIRS CACHE)
unset(GeographicLib_LIBRARIES CACHE)

# Find headers + library in system paths
find_path(GeographicLib_INCLUDE_DIRS
  NAMES GeographicLib/Config.h
  PATH_SUFFIXES include
)
find_library(GeographicLib_LIBRARIES
  NAMES Geographic GeographicLib
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GeographicLib
  REQUIRED_VARS GeographicLib_INCLUDE_DIRS GeographicLib_LIBRARIES
)

set(GTSAM_HAVE_GEOGRAPHICLIB ${GeographicLib_FOUND})

if(GeographicLib_FOUND AND NOT TARGET GeographicLib::GeographicLib)
  add_library(GeographicLib::GeographicLib UNKNOWN IMPORTED)
  set_target_properties(GeographicLib::GeographicLib PROPERTIES
    IMPORTED_LOCATION             "${GeographicLib_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${GeographicLib_INCLUDE_DIRS}"
  )
endif()
