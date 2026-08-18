"""Mesh math shared by the check modules.

trimesh provides topology and adjacency, and robust_laplacian provides the
cotangent Laplacian. This module implements the pieces with no dependable library
equivalent: UV atlas rasterization, triangle-triangle overlap, bow-tie vertex
detection, and polygon-soup topology.
"""

import functools

import numpy as np
import robust_laplacian
import scipy.sparse
import scipy.sparse.csgraph
import trimesh
import trimesh.graph
import trimesh.grouping
import trimesh.transformations


class Surface:
    """Derived topology for one mesh, computed on demand and cached.

    The class evaluates topology on a positionally welded copy. glTF splits
    vertices at UV and normal seams, which would otherwise appear as holes and
    extra shells.
    Welding preserves triangle and corner order, so per-corner attributes such
    as UVs stay addressable by the original slot.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, weld_tol: float):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.weld_tol = weld_tol

    @functools.cached_property
    def _weld(self) -> tuple[np.ndarray, np.ndarray]:
        digits = round(-np.log10(self.weld_tol))
        unique, inverse = trimesh.grouping.unique_rows(self.vertices, digits=digits)
        return self.vertices[unique], np.asarray(inverse)

    @property
    def welded_vertices(self) -> np.ndarray:
        return self._weld[0]

    @property
    def weld_map(self) -> np.ndarray:
        """Original vertex index -> welded vertex index."""
        return self._weld[1]

    @property
    def welded_triangles(self) -> np.ndarray:
        return self.weld_map[self.triangles]

    @functools.cached_property
    def mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.welded_vertices,
            faces=self.welded_triangles,
            process=False,
            validate=False,
        )

    @functools.cached_property
    def edge_uses(self) -> np.ndarray:
        """Triangles incident to each entry of `mesh.edges_unique`."""
        return np.bincount(
            self.mesh.edges_unique_inverse, minlength=len(self.mesh.edges_unique)
        )

    @property
    def boundary_edges(self) -> np.ndarray:
        return self.mesh.edges_unique[self.edge_uses == 1]

    @property
    def nonmanifold_edges(self) -> np.ndarray:
        return self.mesh.edges_unique[self.edge_uses > 2]

    @functools.cached_property
    def vertex_components(self) -> np.ndarray:
        return trimesh.graph.connected_component_labels(
            self.mesh.edges_unique, node_count=len(self.welded_vertices)
        )

    @property
    def triangle_components(self) -> np.ndarray:
        return self.vertex_components[self.welded_triangles[:, 0]]

    @property
    def corners(self) -> np.ndarray:
        """Triangle corner positions, (m, 3, 3)."""
        return self.mesh.triangles

    @property
    def face_normals(self) -> np.ndarray:
        return self.mesh.face_normals

    @property
    def areas(self) -> np.ndarray:
        return self.mesh.area_faces

    @property
    def corner_angles(self) -> np.ndarray:
        return self.mesh.face_angles


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Volume implied by triangle winding; negative means inward-facing."""
    return float(trimesh.triangles.mass_properties(vertices[triangles])["volume"])


def interior_defects(surface: Surface) -> np.ndarray:
    """Angle defect at vertices away from any boundary."""
    interior = np.ones(len(surface.welded_vertices), dtype=bool)
    interior[surface.boundary_edges.ravel()] = False
    return surface.mesh.vertex_defects[interior]


def mean_curvature(surface: Surface) -> np.ndarray:
    """Discrete mean curvature per welded vertex, from the cotangent Laplacian.

    Half the magnitude of the Laplacian of position: for a sphere of radius R
    this is 1/R.
    """
    laplacian, mass = robust_laplacian.mesh_laplacian(
        surface.welded_vertices, surface.welded_triangles
    )
    lumped = np.asarray(mass.diagonal())
    flow = (laplacian @ surface.welded_vertices) / np.where(lumped == 0, 1.0, lumped)[
        :, None
    ]
    return np.linalg.norm(flow, axis=1) / 2.0


class SharedEdges:
    """Triangle pairs that share an edge, with each side's corner slots.

    `left_slots[:, k]` and `right_slots[:, k]` address the same shared-edge vertex
    in both triangles. These slots support comparison of per-corner attributes
    such as UVs.
    """

    def __init__(self, left, right, left_slots, right_slots):
        self.left = left
        self.right = right
        self.left_slots = left_slots
        self.right_slots = right_slots

    def __len__(self) -> int:
        return len(self.left)


def shared_edges(surface: Surface) -> SharedEdges:
    adjacency = surface.mesh.face_adjacency
    if not len(adjacency):
        empty = np.zeros(0, dtype=np.int64)
        return SharedEdges(empty, empty, empty.reshape(0, 2), empty.reshape(0, 2))

    left, right = adjacency[:, 0], adjacency[:, 1]
    endpoints = surface.mesh.face_adjacency_edges
    tris = surface.welded_triangles
    slots = [
        (
            (tris[left] == endpoints[:, k, None]).argmax(axis=1),
            (tris[right] == endpoints[:, k, None]).argmax(axis=1),
        )
        for k in (0, 1)
    ]
    return SharedEdges(
        left,
        right,
        np.stack([slots[0][0], slots[1][0]], axis=1),
        np.stack([slots[0][1], slots[1][1]], axis=1),
    )


def bowtie_vertices(surface: Surface) -> np.ndarray:
    """Vertices whose incident triangles form more than one connected fan."""
    tris = surface.welded_triangles
    if not len(tris):
        return np.zeros(0, dtype=np.int64)

    adjacency = shared_edges(surface)
    rows = np.concatenate(
        [adjacency.left * 3 + adjacency.left_slots[:, k] for k in (0, 1)]
    )
    cols = np.concatenate(
        [adjacency.right * 3 + adjacency.right_slots[:, k] for k in (0, 1)]
    )
    nodes = len(tris) * 3
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(nodes, nodes)
    )
    _, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)

    fans = np.unique(np.stack([tris.ravel(), labels], axis=1), axis=0)
    vertices, counts = np.unique(fans[:, 0], return_counts=True)
    return vertices[counts > 1]


def polygon_edges(
    face_counts: np.ndarray, face_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected polygon edges and how many polygons use each."""
    starts = np.repeat(np.concatenate([[0], np.cumsum(face_counts)[:-1]]), face_counts)
    within = np.arange(len(face_indices)) - starts
    following = starts + (within + 1) % np.repeat(face_counts, face_counts)
    edges = np.stack([face_indices, face_indices[following]], axis=1)
    return np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)


def median_edge_length(surface: Surface) -> np.ndarray:
    """Median length of the edges incident to each welded vertex."""
    edges = surface.mesh.edges_unique
    verts = surface.welded_vertices
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)

    owners = np.concatenate([edges[:, 0], edges[:, 1]])
    lengths = np.tile(lengths, 2)
    order = np.lexsort((lengths, owners))
    owners, lengths = owners[order], lengths[order]

    index = np.arange(len(verts))
    start = np.searchsorted(owners, index, "left")
    end = np.searchsorted(owners, index, "right")
    middle = np.clip((start + end) // 2, 0, max(len(lengths) - 1, 0))
    return np.where(end > start, lengths[middle], 0.0)


def rasterize_uv(
    uv_triangles: np.ndarray, resolution: int, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact triangle coverage of a UV atlas.

    Returns the number of triangles covering each texel and the label of one
    covering triangle per texel (-1 where empty). Grouping triangles by
    bounding-box size keeps the point-in-triangle test vectorized.
    """
    counts = np.zeros((resolution, resolution), dtype=np.int32)
    painted = np.full((resolution, resolution), -1, dtype=np.int64)
    if not len(uv_triangles):
        return counts, painted

    corners = _counterclockwise(np.clip(uv_triangles, 0.0, 1.0) * resolution)
    low = np.floor(corners.min(axis=1)).astype(np.int64)
    span = np.maximum(np.ceil(corners.max(axis=1)).astype(np.int64) - low, 1).max(axis=1)
    bucket = 1 << np.ceil(np.log2(span)).astype(np.int64)

    for size in np.unique(bucket):
        selected = np.nonzero(bucket == size)[0]
        grid = np.stack(
            np.meshgrid(np.arange(size), np.arange(size), indexing="xy"), axis=-1
        ).reshape(-1, 2)
        texels = low[selected][:, None, :] + grid[None, :, :]
        inside = _inside_triangle(corners[selected], texels + 0.5)
        inside &= ((texels >= 0) & (texels < resolution)).all(axis=2)

        rows, cols = np.nonzero(inside)
        x, y = texels[rows, cols, 0], texels[rows, cols, 1]
        np.add.at(counts, (y, x), 1)
        painted[y, x] = labels[selected[rows]]
    return counts, painted


def _counterclockwise(triangles: np.ndarray) -> np.ndarray:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    clockwise = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0] < 0
    return np.where(clockwise[:, None, None], triangles[:, ::-1], triangles)


def _inside_triangle(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Point-in-triangle for (n, 3, 2) triangles and (n, k, 2) points.

    Counter-clockwise triangles only. Points exactly on an edge belong to the
    triangle on its top-left side, so adjacent triangles tile without covering
    a shared texel twice.
    """
    inside = np.ones(points.shape[:2], dtype=bool)
    for i in range(3):
        origin = triangles[:, i][:, None, :]
        edge = triangles[:, (i + 1) % 3] - triangles[:, i]
        offset = points - origin
        cross = edge[:, None, 0] * offset[..., 1] - edge[:, None, 1] * offset[..., 0]
        top_left = (edge[:, 1] > 0) | ((edge[:, 1] == 0) & (edge[:, 0] < 0))
        inside &= (cross > 0) | ((cross == 0) & top_left[:, None])
    return inside


def candidate_pairs(surface: Surface) -> np.ndarray:
    """Triangle pairs with overlapping bounds that share no vertex."""
    tree = surface.mesh.triangles_tree
    tris = surface.welded_triangles
    corners = surface.mesh.triangles
    bounds = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
    pairs = []
    for i, box in enumerate(bounds):
        own = set(tris[i].tolist())
        for j in tree.intersection(box):
            if j > i and not own & set(tris[j].tolist()):
                pairs.append((i, j))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def triangles_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Möller's triangle-triangle overlap test, vectorized over pairs.

    Coplanar pairs take a two-dimensional branch that reports positive-area
    overlap: duplicated faces and crossing triangles within one flat panel
    count, triangles that merely share an edge or a corner do not.
    """
    normal_a, dist_a = _plane_distances(a, b)
    normal_b, dist_b = _plane_distances(b, a)
    hits = _straddles(dist_a) & _straddles(dist_b)
    if hits.any():
        axis = np.argmax(np.abs(np.cross(normal_a, normal_b)), axis=1)
        lo_a, hi_a = _plane_interval(a, dist_b, axis)
        lo_b, hi_b = _plane_interval(b, dist_a, axis)
        hits &= (lo_a <= hi_b) & (lo_b <= hi_a)

    coplanar = (dist_a == 0).all(axis=1) & (dist_b == 0).all(axis=1)
    for index in np.nonzero(coplanar)[0]:
        hits[index] = _coplanar_overlap(a[index], b[index], normal_a[index])
    return hits


def _coplanar_overlap(first: np.ndarray, second: np.ndarray, normal: np.ndarray) -> bool:
    """Positive-area overlap of two triangles in the same plane.

    Overlap exists exactly when an edge of one properly crosses an edge of the
    other, or one triangle's vertex or centroid lies strictly inside the other.
    The centroid case catches exact duplicates because their vertices sit on
    each other's boundaries rather than inside them.
    """
    scale = np.abs(normal).max()
    if scale == 0.0:
        return False
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    p, q = first[:, keep], second[:, keep]

    for i in range(3):
        for j in range(3):
            if _proper_crossing(p[i], p[(i + 1) % 3], q[j], q[(j + 1) % 3]):
                return True
    inside_q = any(_strictly_inside(point, q) for point in [*p, p.mean(axis=0)])
    inside_p = any(_strictly_inside(point, p) for point in [*q, q.mean(axis=0)])
    return inside_q or inside_p


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _proper_crossing(a, b, c, d) -> bool:
    """Open-segment crossing: touching at an endpoint does not count."""
    first = _orient(a, b, c) * _orient(a, b, d)
    second = _orient(c, d, a) * _orient(c, d, b)
    return first < 0 and second < 0


def _strictly_inside(point: np.ndarray, triangle: np.ndarray) -> bool:
    signs = [
        _orient(triangle[i], triangle[(i + 1) % 3], point) for i in range(3)
    ]
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _plane_distances(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Plane normal of `source`, and signed distances of `target`'s corners."""
    normal = np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0])
    offset = -np.einsum("ij,ij->i", normal, source[:, 0])
    dist = np.einsum("ijk,ik->ij", target, normal) + offset[:, None]
    scale = np.maximum(np.abs(dist).max(axis=1), 1e-30)
    return normal, np.where(np.abs(dist) / scale[:, None] < 1e-12, 0.0, dist)


def _straddles(dist: np.ndarray) -> np.ndarray:
    return ~(dist > 0).all(axis=1) & ~(dist < 0).all(axis=1) & ~(dist == 0).all(axis=1)


def _plane_interval(
    tri: np.ndarray, dist: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Span of a triangle's plane crossing, projected onto one coordinate axis."""
    rows = np.arange(len(tri))[:, None]
    projected = tri[rows, np.arange(3)[None, :], axis[:, None]]
    lo = np.full(len(tri), np.inf)
    hi = np.full(len(tri), -np.inf)
    for i in range(3):
        j = (i + 1) % 3
        di, dj = dist[:, i], dist[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            crossed = projected[:, i] + (projected[:, j] - projected[:, i]) * di / (
                di - dj
            )
        for value, mask in ((crossed, di * dj < 0), (projected[:, i], di == 0)):
            lo = np.where(mask, np.minimum(lo, value), lo)
            hi = np.where(mask, np.maximum(hi, value), hi)
    return lo, hi


def surface_points(mesh: trimesh.Trimesh, limit: int) -> np.ndarray:
    """Points spread over a mesh's surface, not just its corners.

    Coarse meshes have few vertices and they sit on the silhouette, so vertices
    alone miss overlaps: two cubes meeting face to face share no nearby pair.
    """
    if not len(mesh.faces):
        return np.asarray(mesh.vertices)
    return subsample(
        np.concatenate([mesh.vertices, mesh.triangles_center]), limit
    )


def penetration_depth(
    a: trimesh.Trimesh, b: trimesh.Trimesh, samples: int
) -> tuple[float, np.ndarray]:
    """Deepest overlap between two meshes, and the point where it occurs.

    Ray parity approximates containment on open meshes. Shallow depths near an
    open boundary are less reliable than deep ones.
    """
    depth, location = 0.0, np.zeros(3)
    for source, target in ((a, b), (b, a)):
        points = surface_points(source, samples)
        if not len(points) or not len(target.faces):
            continue
        inside = target.ray.contains_points(points)
        if not inside.any():
            continue
        _, distance, _ = trimesh.proximity.closest_point(target, points[inside])
        best = int(np.argmax(distance))
        if distance[best] > depth:
            depth, location = float(distance[best]), points[inside][best]
    return depth, location


def subsample(points: np.ndarray, limit: int) -> np.ndarray:
    points = np.asarray(points)
    if len(points) <= limit:
        return points
    return points[(np.arange(limit) * (len(points) / limit)).astype(int)]


def fibonacci_sphere(count: int) -> np.ndarray:
    """Roughly uniform directions on the unit sphere."""
    i = np.arange(count) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / count)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def occlusion(
    mesh: trimesh.Trimesh, points: np.ndarray, normals: np.ndarray, directions: int
) -> np.ndarray:
    """Cosine-weighted blocked fraction of the hemisphere at each point."""
    sphere = fibonacci_sphere(directions)
    cosines = normals @ sphere.T
    rows, cols = np.nonzero(cosines > 0.0)
    if not len(rows):
        return np.zeros(len(points))

    scale = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) or 1.0
    origins = points[rows] + normals[rows] * 1e-5 * scale
    blocked = mesh.ray.intersects_any(origins, sphere[cols])

    weight = cosines[rows, cols]
    total, hit = np.zeros(len(points)), np.zeros(len(points))
    np.add.at(total, rows, weight)
    np.add.at(hit, rows, weight * blocked)
    return hit / np.where(total == 0, 1.0, total)


def convex_hull(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    return np.asarray(hull.vertices), np.asarray(hull.faces)


def box_mesh(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box = trimesh.creation.box(extents=bounds[1] - bounds[0])
    return np.asarray(box.vertices) + bounds.mean(axis=0), np.asarray(box.faces)


def sphere_mesh(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    return np.asarray(sphere.vertices) + center, np.asarray(sphere.faces)
