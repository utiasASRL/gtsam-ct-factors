###############################################################################
# Find boost

# To change the path for boost, you will need to set:
# BOOST_ROOT: path to install prefix for boost
# Boost_NO_SYSTEM_PATHS: set to true to keep the find script from ignoring BOOST_ROOT

if(MSVC)
    # By default, boost only builds static libraries on windows
    set(Boost_USE_STATIC_LIBS ON)  # only find static libs
    # If we ever reset above on windows and, ...
    # If we use Boost shared libs, disable auto linking.
    # Some libraries, at least Boost Program Options, rely on this to export DLL symbols.
    if(NOT Boost_USE_STATIC_LIBS)
        list_append_cache(GTSAM_COMPILE_DEFINITIONS_PUBLIC BOOST_ALL_NO_LIB BOOST_ALL_DYN_LINK)
    endif()
    # Virtual memory range for PCH exceeded on VS2015
    if(MSVC_VERSION LESS 1910) # older than VS2017
      list_append_cache(GTSAM_COMPILE_OPTIONS_PRIVATE -Zm295)
    endif()
endif()

# Prefer Boost's CMake package over FindBoost when available (CMake >= 3.30)
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(BOOST_FIND_MINIMUM_VERSION 1.70)
# Do NOT include 'system' as a required component; it's header-only in modern Boost
set(BOOST_FIND_MINIMUM_COMPONENTS serialization filesystem thread program_options date_time timer chrono regex)

# Prefer the Boost CMake package (CONFIG); fall back to the module if needed.
# Make 'system' optional to support Boost >= 1.69 where it is header-only.
find_package(Boost ${BOOST_FIND_MINIMUM_VERSION} REQUIRED
             COMPONENTS ${BOOST_FIND_MINIMUM_COMPONENTS}
             OPTIONAL_COMPONENTS system CONFIG)

# Verify required imported targets exist (works for both BoostConfig and FindBoost)
foreach(_t IN ITEMS Boost::serialization Boost::filesystem Boost::thread Boost::date_time)
  if(NOT TARGET ${_t})
    message(FATAL_ERROR "Missing required Boost component target: ${_t}. Please install/upgrade Boost or set BOOST_ROOT/Boost_DIR correctly.")
  endif()
endforeach()

option(GTSAM_DISABLE_NEW_TIMERS "Disables using Boost.chrono for timing" OFF)

set(GTSAM_BOOST_LIBRARIES
  Boost::serialization
  Boost::filesystem
  Boost::thread
  Boost::date_time
  Boost::regex
)

# Link Boost::system only when the target exists (older Boost providing a stub library)
if(TARGET Boost::system)
  list(APPEND GTSAM_BOOST_LIBRARIES Boost::system)
endif()

if(GTSAM_DISABLE_NEW_TIMERS)
  message("WARNING:  GTSAM timing instrumentation manually disabled")
  list_append_cache(GTSAM_COMPILE_DEFINITIONS_PUBLIC DGTSAM_DISABLE_NEW_TIMERS)
else()
  # Prefer linking Boost::timer and Boost::chrono if available as imported targets
  if(TARGET Boost::timer AND TARGET Boost::chrono)
    list(APPEND GTSAM_BOOST_LIBRARIES Boost::timer Boost::chrono)
  else()
    # Fall back: when using the header-only timer, librt is needed on Linux only
    if(UNIX AND NOT APPLE)
      list(APPEND GTSAM_BOOST_LIBRARIES rt)
      message("WARNING:  Using header-only Boost timer; adding -lrt on Linux.")
    else()
      message("WARNING:  Using header-only Boost timer; no extra libs required on this platform.")
    endif()
  endif()
endif()
