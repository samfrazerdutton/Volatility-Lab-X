// SPDX-License-Identifier: MIT
#pragma once
/// \file dependency_graph.hpp
/// \brief A generic dirty-propagation DAG: what needs recomputing when one
///        thing changes.
///
/// ## The problem this solves
///
/// A quote tick touches one expiry. That dirties its own slice's
/// calibration (`calibration/incremental.hpp`), which dirties everything
/// built from the surface the slice belongs to -- the differential engine,
/// the uncertainty estimate, every portfolio position's Greeks -- but it
/// does *not* dirty an unrelated underlier's surface, or a slice three
/// expiries away. A system that cannot tell the difference recomputes
/// everything on every tick, which is exactly the waste incremental
/// calibration exists to avoid one level down; this is the same idea one
/// level up, generalised to an arbitrary web of "what depends on what".
///
/// ## Why this is a plain bookkeeping structure, not a scheduler
///
/// This graph tracks *labels and dirty flags*; it does not call anything.
/// Each real computation (recalibrate a slice, recompute a differential,
/// re-run an uncertainty estimate) stays exactly where it already lives, in
/// its own module with its own tests. A caller asks this graph "what is
/// dirty, in an order safe to recompute in", does the actual work itself in
/// that order, and calls `mark_clean` on each node as it finishes. Folding
/// the recompute calls themselves into the graph would couple a generic,
/// easily-tested primitive to every concrete computation in the codebase --
/// exactly the kind of unnecessary abstraction the project avoids elsewhere.
///
/// ## Why cycles are impossible by construction
///
/// `add_node` requires every dependency to already be a node in the graph --
/// there is no way to add an edge to a node that does not exist yet. That
/// means insertion order is *always* a valid topological order (every
/// node's dependencies were added, and therefore appear earlier in
/// insertion order, before the node itself), so `dirty_nodes_in_order`
/// needs nothing more than a single filtering pass over the node list in
/// insertion order -- no cycle detection, no separate topological sort to
/// get right or to test.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vl {

using NodeId = std::uint32_t;

/// Sentinel for "no such node" / "not found".
inline constexpr NodeId kInvalidNodeId = static_cast<NodeId>(-1);

class DependencyGraph {
  public:
    /// Register a new node depending on zero or more existing nodes.
    /// `depends_on` must contain only ids already returned by a previous
    /// `add_node` call on this graph -- see the file comment for why that is
    /// what makes a cycle impossible to construct.
    ///
    /// A new node starts dirty: nothing has been computed for it yet, so
    /// there is nothing clean to claim.
    [[nodiscard]] NodeId add_node(std::string label, std::span<const NodeId> depends_on = {});

    /// Mark one node's own data changed, and transitively every node that
    /// depends on it, directly or indirectly. Idempotent: marking an
    /// already-dirty node (or re-dirtying a subtree that is already dirty)
    /// changes nothing further.
    void mark_dirty(NodeId id);

    /// Mark exactly this one node clean, after its own value has been
    /// recomputed from its (by then clean) dependencies. Does *not* touch
    /// its dependents -- a node recomputed itself, not on their behalf, so
    /// they stay dirty until recomputed too.
    ///
    /// Contract: every one of `id`'s own dependencies must already be clean
    /// when this is called -- calling it from dirty inputs means whatever
    /// was computed is stale. Processing `dirty_nodes_in_order()` in order,
    /// recomputing and marking each one clean before moving to the next,
    /// always satisfies this. Debug builds assert it.
    void mark_clean(NodeId id);

    [[nodiscard]] bool is_dirty(NodeId id) const;

    /// Total number of registered nodes.
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }

    [[nodiscard]] const std::string& label(NodeId id) const;
    [[nodiscard]] std::span<const NodeId> dependencies_of(NodeId id) const;
    [[nodiscard]] std::span<const NodeId> dependents_of(NodeId id) const;

    /// Every currently-dirty node, in insertion order -- which, per the file
    /// comment, is always a valid order to recompute them in: a node never
    /// appears before any of its own dependencies.
    [[nodiscard]] std::vector<NodeId> dirty_nodes_in_order() const;

    /// Convenience: true if every node is clean.
    [[nodiscard]] bool all_clean() const noexcept;

  private:
    struct Node {
        std::string label;
        std::vector<NodeId> dependencies;
        std::vector<NodeId> dependents;
        bool dirty = true;
    };
    std::vector<Node> nodes_;
};

}  // namespace vl
