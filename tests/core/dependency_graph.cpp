// SPDX-License-Identifier: MIT
/// Validates the dirty-propagation dependency graph: that a new node starts
/// dirty, that dirtying one node propagates to every transitive dependent
/// and no further, that mark_clean touches only the node it is called on,
/// and that insertion order is always a safe recompute order.

#include "vl_test_support.hpp"

#include "volatility_lab/core/dependency_graph.hpp"

#include <algorithm>
#include <array>

using namespace vl;

// ---------------------------------------------------------------------------
// Basic bookkeeping
// ---------------------------------------------------------------------------

TEST(DependencyGraph, NewNodeStartsDirty) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    EXPECT_TRUE(g.is_dirty(a));
    EXPECT_FALSE(g.all_clean());
}

TEST(DependencyGraph, LabelsAndEdgesAreReportedBack) {
    DependencyGraph g;
    const NodeId a = g.add_node("quotes[T=0.25]");
    const NodeId b = g.add_node("slice[T=0.25]", std::array{a});
    EXPECT_EQ(g.label(a), "quotes[T=0.25]");
    EXPECT_EQ(g.label(b), "slice[T=0.25]");
    ASSERT_EQ(g.dependencies_of(b).size(), 1u);
    EXPECT_EQ(g.dependencies_of(b)[0], a);
    ASSERT_EQ(g.dependents_of(a).size(), 1u);
    EXPECT_EQ(g.dependents_of(a)[0], b);
}

TEST(DependencyGraph, EmptyGraphIsVacuouslyAllClean) {
    DependencyGraph g;
    EXPECT_TRUE(g.all_clean());
    EXPECT_TRUE(g.dirty_nodes_in_order().empty());
}

// ---------------------------------------------------------------------------
// Propagation
// ---------------------------------------------------------------------------

TEST(DependencyGraph, DirtyingASourcePropagatesThroughAChain) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    const NodeId c = g.add_node("c", std::array{b});
    // All three start dirty (every node does); clean them all first so the
    // propagation from a fresh mark_dirty(a) is unambiguous.
    g.mark_clean(a);
    g.mark_clean(b);
    g.mark_clean(c);
    ASSERT_TRUE(g.all_clean());

    g.mark_dirty(a);
    EXPECT_TRUE(g.is_dirty(a));
    EXPECT_TRUE(g.is_dirty(b));
    EXPECT_TRUE(g.is_dirty(c));
}

TEST(DependencyGraph, DirtyingAMidChainNodeDoesNotDirtyItsOwnDependencies) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    const NodeId c = g.add_node("c", std::array{b});
    g.mark_clean(a);
    g.mark_clean(b);
    g.mark_clean(c);

    g.mark_dirty(b);
    EXPECT_FALSE(g.is_dirty(a)) << "dirtying b must not dirty what b depends on";
    EXPECT_TRUE(g.is_dirty(b));
    EXPECT_TRUE(g.is_dirty(c));
}

TEST(DependencyGraph, DiamondPropagatesThroughBothBranchesToTheSink) {
    //      a
    //     / \
    //    b   c
    //     \ /
    //      d
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    const NodeId c = g.add_node("c", std::array{a});
    const NodeId d = g.add_node("d", std::array{b, c});
    for (NodeId n : {a, b, c, d}) g.mark_clean(n);

    g.mark_dirty(a);
    EXPECT_TRUE(g.is_dirty(b));
    EXPECT_TRUE(g.is_dirty(c));
    EXPECT_TRUE(g.is_dirty(d)) << "d depends on both branches and must be dirtied via either";
}

TEST(DependencyGraph, UnrelatedSiblingIsNotDisturbed) {
    // Two independent chains sharing nothing; dirtying one must leave the
    // other completely untouched -- this is the whole point of the graph
    // over "recompute everything".
    DependencyGraph g;
    const NodeId a1 = g.add_node("a1");
    const NodeId b1 = g.add_node("b1", std::array{a1});
    const NodeId a2 = g.add_node("a2");
    const NodeId b2 = g.add_node("b2", std::array{a2});
    for (NodeId n : {a1, b1, a2, b2}) g.mark_clean(n);

    g.mark_dirty(a1);
    EXPECT_TRUE(g.is_dirty(a1));
    EXPECT_TRUE(g.is_dirty(b1));
    EXPECT_FALSE(g.is_dirty(a2));
    EXPECT_FALSE(g.is_dirty(b2));
}

TEST(DependencyGraph, MarkDirtyIsIdempotent) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    g.mark_clean(a);
    g.mark_clean(b);

    g.mark_dirty(a);
    g.mark_dirty(a);  // second call: no-op, nothing to re-propagate
    EXPECT_TRUE(g.is_dirty(a));
    EXPECT_TRUE(g.is_dirty(b));
}

// ---------------------------------------------------------------------------
// mark_clean: exactly one node, never its dependents
// ---------------------------------------------------------------------------

TEST(DependencyGraph, MarkCleanTouchesOnlyItsOwnNode) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    g.mark_clean(a);
    g.mark_clean(b);
    g.mark_dirty(a);
    ASSERT_TRUE(g.is_dirty(a));
    ASSERT_TRUE(g.is_dirty(b));

    g.mark_clean(a);
    EXPECT_FALSE(g.is_dirty(a));
    EXPECT_TRUE(g.is_dirty(b)) << "b was never recomputed and must stay dirty";
}

TEST(DependencyGraph, ReDirtyingAfterAPartialCleanCorrectlyReDirtiesTheCleanedDependent) {
    // a -> b. Dirty both, clean a, clean b (a valid full recompute pass).
    // Then a changes again: b must become dirty again too, even though
    // mark_dirty short-circuits on an already-dirty node -- it must not
    // short-circuit here because a was clean at the time of the second call.
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    g.mark_dirty(a);  // a, b already dirty from construction; explicit for clarity
    g.mark_clean(a);
    g.mark_clean(b);
    ASSERT_TRUE(g.all_clean());

    g.mark_dirty(a);
    EXPECT_TRUE(g.is_dirty(a));
    EXPECT_TRUE(g.is_dirty(b));
}

// ---------------------------------------------------------------------------
// Recompute ordering
// ---------------------------------------------------------------------------

TEST(DependencyGraph, DirtyNodesInOrderNeverListsADependentBeforeItsDependency) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    const NodeId c = g.add_node("c", std::array{b});
    const NodeId d = g.add_node("d", std::array{a});
    // All four are dirty from construction.
    const auto order = g.dirty_nodes_in_order();
    ASSERT_EQ(order.size(), 4u);

    auto position_of = [&](NodeId id) {
        return static_cast<std::size_t>(std::find(order.begin(), order.end(), id) -
                                        order.begin());
    };
    EXPECT_LT(position_of(a), position_of(b));
    EXPECT_LT(position_of(b), position_of(c));
    EXPECT_LT(position_of(a), position_of(d));
}

TEST(DependencyGraph, DirtyNodesInOrderOmitsCleanNodes) {
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    const NodeId c = g.add_node("c", std::array{a});
    g.mark_clean(a);
    g.mark_clean(b);
    g.mark_clean(c);
    g.mark_dirty(a);
    g.mark_clean(a);  // a itself is clean again; only recompute a's siblings' staleness, not a

    // b and c are still dirty from the earlier propagation (never recomputed).
    const auto order = g.dirty_nodes_in_order();
    EXPECT_EQ(order.size(), 2u);
    EXPECT_FALSE(std::find(order.begin(), order.end(), a) != order.end());
}

TEST(DependencyGraph, ProcessingDirtyNodesInOrderAndCleaningEachSatisfiesTheMarkCleanContract) {
    // The realistic usage pattern this graph is designed for: walk the
    // dirty list in order, "recompute" (here, nothing but the bookkeeping)
    // each one, and mark it clean immediately -- this must never trip the
    // "dependency still dirty" debug assertion in mark_clean, for an
    // arbitrarily-shaped DAG.
    DependencyGraph g;
    const NodeId a = g.add_node("a");
    const NodeId b = g.add_node("b", std::array{a});
    const NodeId c = g.add_node("c", std::array{a});
    const NodeId d = g.add_node("d", std::array{b, c});
    (void)d;
    for (const NodeId id : g.dirty_nodes_in_order()) {
        g.mark_clean(id);  // would assert if a dependency were still dirty
    }
    EXPECT_TRUE(g.all_clean());
}
