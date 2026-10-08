// SPDX-License-Identifier: MIT
#include "volatility_lab/core/dependency_graph.hpp"

#include <cassert>
#include <vector>

namespace vl {

NodeId DependencyGraph::add_node(std::string label, std::span<const NodeId> depends_on) {
    const NodeId id = static_cast<NodeId>(nodes_.size());
    Node node;
    node.label = std::move(label);
    node.dependencies.assign(depends_on.begin(), depends_on.end());
    nodes_.push_back(std::move(node));

    for (NodeId dep : depends_on) {
        assert(dep < id && "a dependency must already exist in the graph");
        nodes_[dep].dependents.push_back(id);
    }
    return id;
}

void DependencyGraph::mark_dirty(NodeId id) {
    if (id >= nodes_.size() || nodes_[id].dirty) return;  // already dirty: nothing to propagate
    // Iterative DFS rather than recursion: the depth is bounded by the
    // length of the longest dependency chain, which is unbounded in
    // principle for a large enough graph, so this should not grow the C++
    // call stack.
    std::vector<NodeId> stack{id};
    while (!stack.empty()) {
        const NodeId cur = stack.back();
        stack.pop_back();
        if (nodes_[cur].dirty) continue;
        nodes_[cur].dirty = true;
        for (NodeId dependent : nodes_[cur].dependents) stack.push_back(dependent);
    }
}

void DependencyGraph::mark_clean(NodeId id) {
    if (id >= nodes_.size()) return;
#ifndef NDEBUG
    // mark_clean asserts its own contract: a node can only correctly be
    // recomputed (and so marked clean) once every one of its own
    // dependencies is already clean. Violating this -- cleaning a node
    // while something it reads from is still stale -- is a caller bug this
    // can catch cheaply in debug builds rather than silently computing a
    // value from stale inputs.
    for (NodeId dep : nodes_[id].dependencies) {
        assert(!nodes_[dep].dirty && "marking a node clean while a dependency is still dirty");
    }
#endif
    nodes_[id].dirty = false;
}

bool DependencyGraph::is_dirty(NodeId id) const {
    return id < nodes_.size() && nodes_[id].dirty;
}

const std::string& DependencyGraph::label(NodeId id) const { return nodes_[id].label; }

std::span<const NodeId> DependencyGraph::dependencies_of(NodeId id) const {
    return nodes_[id].dependencies;
}

std::span<const NodeId> DependencyGraph::dependents_of(NodeId id) const {
    return nodes_[id].dependents;
}

std::vector<NodeId> DependencyGraph::dirty_nodes_in_order() const {
    std::vector<NodeId> out;
    out.reserve(nodes_.size());
    for (NodeId i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].dirty) out.push_back(i);
    }
    return out;
}

bool DependencyGraph::all_clean() const noexcept {
    for (const auto& n : nodes_) {
        if (n.dirty) return false;
    }
    return true;
}

}  // namespace vl
