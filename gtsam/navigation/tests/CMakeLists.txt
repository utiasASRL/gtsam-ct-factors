set(test_link_libraries gtsam)
set(tests_excluded "")

if(TARGET GeographicLib::GeographicLib)
  list(APPEND test_link_libraries GeographicLib::GeographicLib)
else()
  list(APPEND tests_excluded
       testGeographicLib.cpp
       testGPSFactor.cpp
       testMagFactor.cpp)
endif()

gtsamaddtestsglob(navigation "test*.cpp" "${tests_excluded}" "${test_link_libraries}")
