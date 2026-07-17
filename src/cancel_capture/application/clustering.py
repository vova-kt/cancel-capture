from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from cancel_capture.models import SearchDocument


@dataclass(frozen=True, slots=True)
class LinkageMerge:
    node_id: int
    left_node_id: int
    right_node_id: int
    distance: float
    size: int


@dataclass(frozen=True, slots=True)
class ClusterGroup:
    node_id: int
    documents: tuple[SearchDocument, ...]

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(document.item_id for document in self.documents)


@dataclass(frozen=True, slots=True)
class DendrogramSegment:
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    merge_node_id: int


@dataclass(frozen=True, slots=True)
class DendrogramGeometry:
    leaf_item_ids: tuple[str, ...]
    segments: tuple[DendrogramSegment, ...]
    max_distance: float


@dataclass(frozen=True, slots=True)
class HierarchicalClustering:
    documents: tuple[SearchDocument, ...]
    merges: tuple[LinkageMerge, ...]

    def cut(self, cluster_count: int) -> tuple[ClusterGroup, ...]:
        document_count = len(self.documents)
        if document_count == 0:
            if cluster_count == 0:
                return ()
            raise ValueError("An empty clustering only supports zero clusters")
        if not 1 <= cluster_count <= document_count:
            raise ValueError(
                f"Cluster count must be between 1 and {document_count}, got {cluster_count}"
            )

        members: dict[int, tuple[int, ...]] = {index: (index,) for index in range(document_count)}
        active = set(range(document_count))
        for merge in self.merges[: document_count - cluster_count]:
            left = members[merge.left_node_id]
            right = members[merge.right_node_id]
            members[merge.node_id] = tuple(sorted((*left, *right)))
            active.remove(merge.left_node_id)
            active.remove(merge.right_node_id)
            active.add(merge.node_id)

        groups = tuple(
            ClusterGroup(
                node_id=node_id,
                documents=tuple(self.documents[index] for index in members[node_id]),
            )
            for node_id in active
        )
        return tuple(sorted(groups, key=lambda group: group.item_ids))

    def dendrogram(self) -> DendrogramGeometry:
        document_count = len(self.documents)
        if document_count == 0:
            return DendrogramGeometry(leaf_item_ids=(), segments=(), max_distance=0.0)

        merges_by_node = {merge.node_id: merge for merge in self.merges}
        root_node_id = self.merges[-1].node_id if self.merges else 0
        leaf_order: list[int] = []
        pending = [root_node_id]
        while pending:
            node_id = pending.pop()
            if node_id < document_count:
                leaf_order.append(node_id)
                continue
            merge = merges_by_node[node_id]
            pending.append(merge.right_node_id)
            pending.append(merge.left_node_id)

        x_positions = {leaf_id: float(index) for index, leaf_id in enumerate(leaf_order)}
        heights = {leaf_id: 0.0 for leaf_id in leaf_order}
        segments: list[DendrogramSegment] = []
        for merge in self.merges:
            left_x = x_positions[merge.left_node_id]
            right_x = x_positions[merge.right_node_id]
            left_height = heights[merge.left_node_id]
            right_height = heights[merge.right_node_id]
            segments.extend(
                (
                    DendrogramSegment(
                        x_start=left_x,
                        y_start=left_height,
                        x_end=left_x,
                        y_end=merge.distance,
                        merge_node_id=merge.node_id,
                    ),
                    DendrogramSegment(
                        x_start=left_x,
                        y_start=merge.distance,
                        x_end=right_x,
                        y_end=merge.distance,
                        merge_node_id=merge.node_id,
                    ),
                    DendrogramSegment(
                        x_start=right_x,
                        y_start=right_height,
                        x_end=right_x,
                        y_end=merge.distance,
                        merge_node_id=merge.node_id,
                    ),
                )
            )
            x_positions[merge.node_id] = (left_x + right_x) / 2.0
            heights[merge.node_id] = merge.distance

        return DendrogramGeometry(
            leaf_item_ids=tuple(self.documents[index].item_id for index in leaf_order),
            segments=tuple(segments),
            max_distance=self.merges[-1].distance if self.merges else 0.0,
        )


_HeapEntry = tuple[float, tuple[int, ...], tuple[int, ...], int, int]


def average_linkage_cosine(
    documents: tuple[SearchDocument, ...],
) -> HierarchicalClustering:
    ordered = tuple(sorted(documents, key=lambda document: document.item_id))
    _validate_documents(ordered)
    document_count = len(ordered)
    if document_count <= 1:
        return HierarchicalClustering(documents=ordered, merges=())

    members: dict[int, tuple[int, ...]] = {index: (index,) for index in range(document_count)}
    sizes = {index: 1 for index in range(document_count)}
    active = set(range(document_count))
    distances: dict[tuple[int, int], float] = {}
    heap: list[_HeapEntry] = []
    for left in range(document_count):
        for right in range(left + 1, document_count):
            distance = _cosine_distance(
                ordered[left].embedding.values,
                ordered[right].embedding.values,
            )
            distances[(left, right)] = distance
            heap.append((distance, members[left], members[right], left, right))
    heapq.heapify(heap)

    merges: list[LinkageMerge] = []
    next_node_id = document_count
    while len(active) > 1:
        distance, _left_key, _right_key, left_node_id, right_node_id = heapq.heappop(heap)
        if left_node_id not in active or right_node_id not in active:
            continue

        left_members = members[left_node_id]
        right_members = members[right_node_id]
        if right_members < left_members:
            left_node_id, right_node_id = right_node_id, left_node_id
            left_members, right_members = right_members, left_members
        left_size = sizes[left_node_id]
        right_size = sizes[right_node_id]
        merged_members = tuple(sorted((*left_members, *right_members)))
        other_nodes = tuple(active - {left_node_id, right_node_id})

        new_distances: list[tuple[int, float]] = []
        for other_node_id in other_nodes:
            left_distance = distances[_node_pair(left_node_id, other_node_id)]
            right_distance = distances[_node_pair(right_node_id, other_node_id)]
            merged_distance = (left_size * left_distance + right_size * right_distance) / (
                left_size + right_size
            )
            new_distances.append((other_node_id, merged_distance))

        merges.append(
            LinkageMerge(
                node_id=next_node_id,
                left_node_id=left_node_id,
                right_node_id=right_node_id,
                distance=distance,
                size=left_size + right_size,
            )
        )
        active.remove(left_node_id)
        active.remove(right_node_id)
        for pair in tuple(distances):
            if left_node_id in pair or right_node_id in pair:
                del distances[pair]
        members[next_node_id] = merged_members
        sizes[next_node_id] = left_size + right_size
        active.add(next_node_id)

        for other_node_id, merged_distance in new_distances:
            pair = _node_pair(next_node_id, other_node_id)
            distances[pair] = merged_distance
            first_node_id, second_node_id = _ordered_by_members(pair[0], pair[1], members)
            heapq.heappush(
                heap,
                (
                    merged_distance,
                    members[first_node_id],
                    members[second_node_id],
                    first_node_id,
                    second_node_id,
                ),
            )
        next_node_id += 1

    return HierarchicalClustering(documents=ordered, merges=tuple(merges))


def _validate_documents(documents: tuple[SearchDocument, ...]) -> None:
    if not documents:
        return
    item_ids = tuple(document.item_id for document in documents)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Clustering documents must have unique item IDs")

    identity = documents[0].embedding.identity
    dimensions = len(documents[0].embedding.values)
    for document in documents[1:]:
        if document.embedding.identity != identity:
            raise ValueError("Cannot cluster embeddings from different provider identities")
        if len(document.embedding.values) != dimensions:
            raise ValueError("Cannot cluster embeddings with different dimensions")


def _cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return 1.0 - min(1.0, max(-1.0, similarity))


def _node_pair(left_node_id: int, right_node_id: int) -> tuple[int, int]:
    return (
        (left_node_id, right_node_id)
        if left_node_id < right_node_id
        else (right_node_id, left_node_id)
    )


def _ordered_by_members(
    left_node_id: int,
    right_node_id: int,
    members: dict[int, tuple[int, ...]],
) -> tuple[int, int]:
    if members[left_node_id] <= members[right_node_id]:
        return left_node_id, right_node_id
    return right_node_id, left_node_id
