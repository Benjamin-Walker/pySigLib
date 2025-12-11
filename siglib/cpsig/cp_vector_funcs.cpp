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

#include "cppch.h"
#include "cpsig.h"
#include "cp_vector_funcs.h"

void call_tensor_vec_mult_add(
    uint64_t n,
    double* a_end,
    double* b_end,
    double* c_start,
    double* z_start,
    uint64_t c_sz
) {
    switch (n) {
    case 1:  return tensor_vec_mult_add_template<1>(a_end, b_end, c_start, z_start, c_sz);
    case 2:  return tensor_vec_mult_add_template<2>(a_end, b_end, c_start, z_start, c_sz);
    case 3:  return tensor_vec_mult_add_template<3>(a_end, b_end, c_start, z_start, c_sz);
    case 4:  return tensor_vec_mult_add_template<4>(a_end, b_end, c_start, z_start, c_sz);
    case 5:  return tensor_vec_mult_add_template<5>(a_end, b_end, c_start, z_start, c_sz);
    case 6:  return tensor_vec_mult_add_template<6>(a_end, b_end, c_start, z_start, c_sz);
    case 7:  return tensor_vec_mult_add_template<7>(a_end, b_end, c_start, z_start, c_sz);
    case 8:  return tensor_vec_mult_add_template<8>(a_end, b_end, c_start, z_start, c_sz);
    case 9:  return tensor_vec_mult_add_template<9>(a_end, b_end, c_start, z_start, c_sz);
    case 10: return tensor_vec_mult_add_template<10>(a_end, b_end, c_start, z_start, c_sz);
    case 11: return tensor_vec_mult_add_template<11>(a_end, b_end, c_start, z_start, c_sz);
    case 12: return tensor_vec_mult_add_template<12>(a_end, b_end, c_start, z_start, c_sz);
    case 13: return tensor_vec_mult_add_template<13>(a_end, b_end, c_start, z_start, c_sz);
    case 14: return tensor_vec_mult_add_template<14>(a_end, b_end, c_start, z_start, c_sz);
    case 15: return tensor_vec_mult_add_template<15>(a_end, b_end, c_start, z_start, c_sz);
    case 16: return tensor_vec_mult_add_template<16>(a_end, b_end, c_start, z_start, c_sz);
    case 17: return tensor_vec_mult_add_template<17>(a_end, b_end, c_start, z_start, c_sz);
    case 18: return tensor_vec_mult_add_template<18>(a_end, b_end, c_start, z_start, c_sz);
    case 19: return tensor_vec_mult_add_template<19>(a_end, b_end, c_start, z_start, c_sz);
    case 20: return tensor_vec_mult_add_template<20>(a_end, b_end, c_start, z_start, c_sz);
    default:
        return tensor_vec_mult_add(a_end, b_end, c_start, z_start, c_sz, n);
    }
}

void call_tensor_vec_mult_assign(
    uint64_t n,
    double* a_end,
    double* b_end,
    double* c_start,
    double* z_start,
    uint64_t c_sz,
    double one_over_level)
{
    switch (n) {
    case 1:  return tensor_vec_mult_assign_template<1>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 2:  return tensor_vec_mult_assign_template<2>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 3:  return tensor_vec_mult_assign_template<3>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 4:  return tensor_vec_mult_assign_template<4>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 5:  return tensor_vec_mult_assign_template<5>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 6:  return tensor_vec_mult_assign_template<6>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 7:  return tensor_vec_mult_assign_template<7>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 8:  return tensor_vec_mult_assign_template<8>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 9:  return tensor_vec_mult_assign_template<9>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 10: return tensor_vec_mult_assign_template<10>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 11: return tensor_vec_mult_assign_template<11>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 12: return tensor_vec_mult_assign_template<12>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 13: return tensor_vec_mult_assign_template<13>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 14: return tensor_vec_mult_assign_template<14>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 15: return tensor_vec_mult_assign_template<15>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 16: return tensor_vec_mult_assign_template<16>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 17: return tensor_vec_mult_assign_template<17>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 18: return tensor_vec_mult_assign_template<18>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 19: return tensor_vec_mult_assign_template<19>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    case 20: return tensor_vec_mult_assign_template<20>(a_end, b_end, c_start, z_start, c_sz, one_over_level);
    default:
        return tensor_vec_mult_assign(a_end, b_end, c_start, z_start, c_sz, n, one_over_level);
    }
}
