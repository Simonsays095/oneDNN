/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_KERNEL_SELECTOR_HPP
#define GEMMSTONE_GUARD_KERNEL_SELECTOR_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"

#include <algorithm>
#include <functional>

GEMMSTONE_NAMESPACE_START

// Basic kernel selection API.
struct StrategyRequirement {
    enum Parameter { UnrollM, UnrollN, WGTileM, WGTileN, WGTileMN, WGM, WGN, WGK, WG } param;
    enum Relation { Equals, AtLeast, AtMost } relation;
    int value;

    StrategyRequirement(Parameter param_, Relation relation_, int value_) : param(param_), relation(relation_), value(value_) {}

    template <typename T> friend StrategyRequirement operator==(Parameter param_, T value_) { return StrategyRequirement(param_, Equals,  int(value_)); }
    template <typename T> friend StrategyRequirement operator<=(Parameter param_, T value_) { return StrategyRequirement(param_, AtMost,  int(value_)); }
    template <typename T> friend StrategyRequirement operator>=(Parameter param_, T value_) { return StrategyRequirement(param_, AtLeast, int(value_)); }

    void transpose();
};

struct MatchParamsBase
{
    kcatalog::Selector selector;
    SizeParams sizes;
    char precisionCExt = 0;
    bool ignoreSizes = false;
    bool ignoreCase = false;
    int stepping = 0;
    int alignment[3] = {0, 0, 0};
    kcatalog::string tags, lateTags;
    int nExtraReqs = 0;
    const StrategyRequirement *extraReqs = nullptr;

    MatchParamsBase() = default;
    MatchParamsBase(ngen::HW hw, bool systolicAvailable, bool isIntegrated, const GEMMProblem &problem);

protected:
    // storage for data in selector
    struct {
        // A/B store single types or conversions like [FO]
        // while C just stores one type (plus null)
        char A[5] = {'\0'};
        char B[5] = {'\0'};
        char C[2] = {'\0'};
    } precisions;
    struct {
        char A[2] = {'\0'};
        char B[2] = {'\0'};
        char C[2] = {'\0'};
    } layouts;
    std::array<char, 12> tagsStorage = {"\0"};
};

struct MatchParams : public MatchParamsBase
{
    MatchParams() = default;
    MatchParams(ngen::HW hw, bool systolicAvailable, bool isIntegrated, const GEMMProblem &problem)
            : MatchParamsBase(hw, systolicAvailable, isIntegrated, problem) {}

    // NOLINTNEXTLINE(bugprone-copy-constructor-init)
    MatchParams(const MatchParams &other) { *this = other; }
    MatchParams &operator=(const MatchParams &other) {
        static_cast<MatchParamsBase &>(*this) = other;

        // Transfers a char * to the new object, but only if
        // it's pointing to the storage object (i.e. not char * literal)
        auto transfer = [&](const char *&value, const char *oldStorage, const char *newStorage) {
            if (value == oldStorage) {
                value = newStorage;
            }
        };

        // Copy selector pointers from the new storage locations
        transfer(selector.precisions[0], &other.precisions.A[0], &precisions.A[0]);
        transfer(selector.precisions[1], &other.precisions.B[0], &precisions.B[0]);
        transfer(selector.precisions[2], &other.precisions.C[0], &precisions.C[0]);

        transfer(selector.layouts[0], &other.layouts.A[0], &layouts.A[0]);
        transfer(selector.layouts[1], &other.layouts.B[0], &layouts.B[0]);
        transfer(selector.layouts[2], &other.layouts.C[0], &layouts.C[0]);

        tags = tagsStorage.data() + (other.tags - other.tagsStorage.data());
        lateTags = tagsStorage.data() + (other.lateTags - other.tagsStorage.data());

        return *this;
    }
};

using SelectionObserver = std::function<void (const kcatalog::Entry *entry, double score, EvaluateAuxOutput aux)>*;

const kcatalog::Entry *select(const kcatalog::Catalog &catalog, const MatchParams &pattern, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver observer = nullptr);
const kcatalog::Entry *select(const kcatalog::Catalog &catalog, int npatterns, const MatchParams *patterns, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver observer = nullptr);

// Extended API for iterating over all matching kernels.
bool matches(const kcatalog::Entry &e, const MatchParams &pattern);
bool lessAligned(int alignA1, int alignB1, int alignA2, int alignB2);

const kcatalog::Entry *lower_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector);
const kcatalog::Entry *upper_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector);

class EntryIterator {
public:
    EntryIterator(const kcatalog::Catalog &catalog_, const MatchParams &pattern_)
        : catalog(catalog_), pattern(pattern_), begin(lower_bound(catalog_, pattern_.selector)), end(upper_bound(catalog_, pattern_.selector)), current(begin) {
        findNextMatch();
    }

    operator bool() const { return current < end; }

    EntryIterator &operator++() {
        ++current;
        findNextMatch();
        return *this;
    }

    EntryIterator operator++(int) {
        auto old = *this;
        operator++();
        return old;
    }

    const kcatalog::Entry &operator*()  const { return  *current; }
    const kcatalog::Entry *operator->() const { return &*current; }

    friend bool operator==(const EntryIterator &i1, const EntryIterator &i2) {
        return (i1.current == i2.current);
    }
    friend bool operator!=(const EntryIterator &i1, const EntryIterator &i2) {
        return !(i1 == i2);
    }

protected:
    const kcatalog::Catalog &catalog;
    MatchParams pattern;
    const kcatalog::Entry *begin, *end, *current;

    void findNextMatch() {
        for (; current < end; current++) {
            if (matches(*current, pattern))
                break;
        }
    }
};

inline EntryIterator match(const kcatalog::Catalog &catalog, const MatchParams &pattern)
{
    return EntryIterator(catalog, pattern);
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
