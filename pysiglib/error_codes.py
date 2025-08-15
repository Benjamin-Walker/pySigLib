# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

CPP_ERR_MSG = {
    1 : "Failed to allocate memory",
    2: "Invalid argument",
    3: "Out of range",
    4: "Runtime error",
    5: "Unknown exception"
}

def err_msg(err_code):
    if err_code < 100000:
        return CPP_ERR_MSG[err_code] + " (" + str(err_code) + ")"
    elif err_code == 100500:
        return "CUDA error: named symbol not found (500). pysiglib: This error may suggest your GPU's compute capability is currently not supported by pysiglib. Please contact the developer."
    return "CUDA error (" + str(err_code - 100000) + ")"
