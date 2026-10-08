// SPDX-License-Identifier: MIT
#include "volatility_lab/calibration/incremental.hpp"

#include <algorithm>
#include <cmath>

namespace vl {

IncrementalUpdateResult apply_incremental_update(const VolSurface& base_surface, double years,
                                                 std::span<const OptionQuote> quotes_for_slice,
                                                 const SviCalibratorConfig& cfg,
                                                 double years_match_tolerance) {
    IncrementalUpdateResult out;
    out.fit = calibrate_svi_slice(quotes_for_slice, cfg);

    const auto expiries = base_surface.expiries();
    const auto it = std::find_if(expiries.begin(), expiries.end(), [&](double t) {
        return std::abs(t - years) <= years_match_tolerance;
    });

    if (it != expiries.end()) {
        // Existing expiry: replace in place, every other slice untouched.
        out.slice_index = static_cast<std::size_t>(it - expiries.begin());
        out.inserted_new_slice = false;
        out.surface = base_surface.with_slice(out.slice_index, SliceVariant(out.fit.params));
        return out;
    }

    // New expiry: insert at the sorted position. with_slices rebuilds the
    // whole slice list (VolSurface has no in-place insert), but this is an
    // O(num_slices) vector splice, not a refit of anything -- the expensive
    // part, calibrate_svi_slice above, still ran exactly once.
    out.inserted_new_slice = true;
    std::vector<SliceVariant> slices(base_surface.slices().begin(), base_surface.slices().end());
    const auto pos = std::lower_bound(expiries.begin(), expiries.end(), years);
    out.slice_index = static_cast<std::size_t>(pos - expiries.begin());
    slices.insert(slices.begin() + static_cast<std::ptrdiff_t>(out.slice_index),
                 SliceVariant(out.fit.params));
    out.surface = base_surface.with_slices(std::move(slices));
    return out;
}

}  // namespace vl
