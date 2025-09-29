set(test_link_libraries gtsam)
set(tests_excluded "")

if(GTSAM_HAVE_GEOGRAPHICLIB)
  list(APPEND test_link_libraries GeographicLib::GeographicLib)
else()
  list(APPEND tests_excluded
       testGeographicLib.cpp
       testGPSFactor.cpp
       testMagFactor.cpp)
endif()

gtsamaddtestsglob(navigation "test*.cpp" "${tests_excluded}" "${test_link_libraries}")
