set(test_link_libraries gtsam)
set(tests_excluded "")


if(TARGET GeographicLib::GeographicLib)
  list(APPEND test_link_libraries GeographicLib::GeographicLib)
else()
  # GeographicLib not available -> skip dependent tests
  list(APPEND tests_excluded
       testGeographicLib.cpp
       testGPSFactor.cpp
       testMagFactor.cpp)
endif()

if(NOT GTSAM_ENABLE_BOOST_SERIALIZATION)
  list(APPEND tests_excluded testSerializationNavigation.cpp)
endif()

gtsamaddtestsglob(navigation "test*.cpp" "${tests_excluded}" "${test_link_libraries}")
igation "test*.cpp" "${tests_excluded}" "${test_link_libraries}")
