/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once
#include <iostream>

//#ifndef __APPLE__
//	#define VEC
//#endif


#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

#define SAFE_CALL(function_call)                                    \
    try {                                                           \
        function_call;                                              \
    }                                                               \
    catch (std::bad_alloc&) {					                    \
		std::cerr << "Failed to allocate memory";                   \
        return 1;                                                   \
    }                                                               \
    catch (std::invalid_argument& e) {                              \
		std::cerr << e.what();					                    \
        return 2;                                                   \
    }                                                               \
	catch (std::out_of_range& e) {			                        \
		std::cerr << e.what();					                    \
		return 3;                                                   \
	}  											                    \
    catch (std::filesystem::filesystem_error& e) {                  \
        std::cerr << e.what();                                      \
        return 4;                                                   \
    }                                                               \
    catch (std::runtime_error& e) {                                 \
        if (std::string(e.what()) == "Could not find basis cache") {\
            std::cerr << e.what();                                  \
            return 5;                                               \
        }                                                           \
        if (std::string(e.what()).rfind("Directory ") == 0) {       \
            std::cerr << e.what();                                  \
            return 6;                                               \
        }                                                           \
        if (std::string(e.what()) == "Failed to get default cache directory.") { \
            std::cerr << e.what();                                  \
            return 7;                                               \
        }                                                           \
        if (std::string(e.what()) == "Unexpected internal error. Cache directory was not set correctly.") {       \
            std::cerr << e.what();                                  \
            return 8;                                               \
        }                                                           \
        if (std::string(e.what()) == "Tried to read an invalid cache file. Cache may have been corrupted.") {       \
            std::cerr << e.what();                                  \
            return 9;                                               \
        }                                                           \
        else {                                                      \
            std::cerr << e.what();                                  \
            return 10;                                               \
        }                                                           \
    }                                                               \
    catch (...) {                                                   \
		std::cerr << "Unknown exception";		                    \
        return 11;                                                   \
    }                                                               \
    return 0;
