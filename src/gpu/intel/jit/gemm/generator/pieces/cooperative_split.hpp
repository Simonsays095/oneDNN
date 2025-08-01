/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#ifndef GEMMSTONE_GUARD_COOPERATIVE_SPLIT_HPP
#define GEMMSTONE_GUARD_COOPERATIVE_SPLIT_HPP

#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gemmstone/driver_info.hpp"

GEMMSTONE_NAMESPACE_START

// Split A/B matrix between threads.
void coopSplit(bool isA, int &splitR, int &splitC, int r, int c, int mnFull, CoopSplit stype, const MatrixAddressing &atype, const GEMMStrategy &strategy);

CoopSplit naturalSplitA(MatrixLayout layout);
CoopSplit naturalSplitB(MatrixLayout layout);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
