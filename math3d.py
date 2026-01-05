#!/usr/bin/env python3
"""
3D Math and Algorithms for PyQt6 JPatch
Includes curve algorithms, surface tessellation, and geometric utilities
"""

import numpy as np
import math
from typing import List, Tuple, Optional

class Vector3D:
    """3D Vector class with common operations"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vector3D(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vector3D(self.x - other, self.y - other, self.z - other)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self):
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other: 'Vector3D') -> float:
        """Dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        """Vector length"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Vector3D':
        """Return normalized vector"""
        length = self.length()
        if length > 0:
            return self / length
        return Vector3D(0, 0, 0)

    def distance_to(self, other: 'Vector3D') -> float:
        """Distance to another vector"""
        return (self - other).length()

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple"""
        return (self.x, self.y, self.z)

    def to_list(self) -> List[float]:
        """Convert to list"""
        return [self.x, self.y, self.z]

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Vector3D':
        """Create from tuple"""
        return cls(t[0], t[1], t[2])

    @classmethod
    def from_list(cls, l: List[float]) -> 'Vector3D':
        """Create from list"""
        return cls(l[0], l[1], l[2])

    def angle_to(self, other: 'Vector3D') -> float:
        """Angle between two vectors in radians"""
        cos_angle = self.dot(other) / (self.length() * other.length())
        cos_angle = clamp(cos_angle, -1.0, 1.0)  # Handle floating point errors
        return math.acos(cos_angle)

    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        """Project this vector onto another vector"""
        if other.length_squared() > 0:
            return other * (self.dot(other) / other.length_squared())
        return Vector3D()

    def __repr__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Matrix4x4:
    """4x4 Matrix class for transformations"""

    def __init__(self, matrix=None):
        if matrix is None:
            # Identity matrix
            self.matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        else:
            self.matrix = [row[:] for row in matrix]

    def __mul__(self, other):
        if isinstance(other, Matrix4x4):
            result = Matrix4x4()
            for i in range(4):
                for j in range(4):
                    result.matrix[i][j] = sum(
                        self.matrix[i][k] * other.matrix[k][j] for k in range(4)
                    )
            return result
        elif isinstance(other, Vector3D):
            # Transform 3D vector (assuming w=1 for points, w=0 for vectors)
            x = (self.matrix[0][0] * other.x + self.matrix[0][1] * other.y +
                 self.matrix[0][2] * other.z + self.matrix[0][3])
            y = (self.matrix[1][0] * other.x + self.matrix[1][1] * other.y +
                 self.matrix[1][2] * other.z + self.matrix[1][3])
            z = (self.matrix[2][0] * other.x + self.matrix[2][1] * other.y +
                 self.matrix[2][2] * other.z + self.matrix[2][3])
            return Vector3D(x, y, z)
        return NotImplemented

    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create translation matrix"""
        matrix = cls()
        matrix.matrix[0][3] = x
        matrix.matrix[1][3] = y
        matrix.matrix[2][3] = z
        return matrix

    @classmethod
    def rotation_x(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around X axis (radians)"""
        matrix = cls()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix.matrix[1][1] = cos_a
        matrix.matrix[1][2] = -sin_a
        matrix.matrix[2][1] = sin_a
        matrix.matrix[2][2] = cos_a
        return matrix

    @classmethod
    def rotation_y(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Y axis (radians)"""
        matrix = cls()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix.matrix[0][0] = cos_a
        matrix.matrix[0][2] = sin_a
        matrix.matrix[2][0] = -sin_a
        matrix.matrix[2][2] = cos_a
        return matrix

    @classmethod
    def rotation_z(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Z axis (radians)"""
        matrix = cls()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix.matrix[0][0] = cos_a
        matrix.matrix[0][1] = -sin_a
        matrix.matrix[1][0] = sin_a
        matrix.matrix[1][1] = cos_a
        return matrix

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> 'Matrix4x4':
        """Create scale matrix"""
        matrix = cls()
        matrix.matrix[0][0] = sx
        matrix.matrix[1][1] = sy
        matrix.matrix[2][2] = sz
        return matrix

    def to_opengl_matrix(self) -> List[float]:
        """Convert to OpenGL column-major format"""
        return [
            self.matrix[0][0], self.matrix[1][0], self.matrix[2][0], self.matrix[3][0],
            self.matrix[0][1], self.matrix[1][1], self.matrix[2][1], self.matrix[3][1],
            self.matrix[0][2], self.matrix[1][2], self.matrix[2][2], self.matrix[3][2],
            self.matrix[0][3], self.matrix[1][3], self.matrix[2][3], self.matrix[3][3]
        ]


class BezierCurve:
    """Bezier curve implementation"""

    def __init__(self, control_points: List[Vector3D]):
        self.control_points = control_points

    def evaluate(self, t: float) -> Vector3D:
        """Evaluate curve at parameter t (0-1)"""
        if len(self.control_points) == 0:
            return Vector3D()
        if len(self.control_points) == 1:
            return self.control_points[0]

        # De Casteljau algorithm
        points = self.control_points[:]
        while len(points) > 1:
            new_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                new_point = p1 + (p2 - p1) * t
                new_points.append(new_point)
            points = new_points

        return points[0]

    def derivative(self, t: float) -> Vector3D:
        """Evaluate first derivative at parameter t"""
        if len(self.control_points) < 2:
            return Vector3D()

        # For Bezier curve, derivative is another Bezier curve with n-1 points
        derivative_points = []
        for i in range(len(self.control_points) - 1):
            p1 = self.control_points[i]
            p2 = self.control_points[i + 1]
            derivative_points.append((p2 - p1) * (len(self.control_points) - 1))

        if len(derivative_points) == 1:
            return derivative_points[0]

        # Evaluate the derivative curve
        deriv_curve = BezierCurve(derivative_points)
        return deriv_curve.evaluate(t)

    def tessellate(self, num_segments: int = 10) -> List[Vector3D]:
        """Tessellate curve into line segments"""
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            points.append(self.evaluate(t))
        return points

    def length(self, num_samples: int = 100) -> float:
        """Approximate curve length"""
        if len(self.control_points) < 2:
            return 0.0

        length = 0.0
        prev_point = self.evaluate(0.0)
        for i in range(1, num_samples + 1):
            t = i / num_samples
            curr_point = self.evaluate(t)
            length += (curr_point - prev_point).length()
            prev_point = curr_point
        return length


class BSplineCurve:
    """B-Spline curve implementation"""

    def __init__(self, control_points: List[Vector3D], degree: int = 3, knots: Optional[List[float]] = None):
        self.control_points = control_points
        self.degree = degree
        self.knots = knots or self._generate_uniform_knots()

    def _generate_uniform_knots(self) -> List[float]:
        """Generate uniform knot vector"""
        n = len(self.control_points) - 1
        p = self.degree
        knots = [0.0] * (p + 1) + list(range(1, n - p + 1)) + [n - p + 1] * (p + 1)
        return [float(k) for k in knots]

    def _cox_de_boor(self, i: int, p: int, t: float) -> float:
        """Cox-de Boor recursion"""
        if p == 0:
            return 1.0 if self.knots[i] <= t < self.knots[i + 1] else 0.0

        left = (t - self.knots[i]) / (self.knots[i + p] - self.knots[i]) if self.knots[i + p] != self.knots[i] else 0.0
        right = (self.knots[i + p + 1] - t) / (self.knots[i + p + 1] - self.knots[i + 1]) if self.knots[i + p + 1] != self.knots[i + 1] else 0.0

        return left * self._cox_de_boor(i, p - 1, t) + right * self._cox_de_boor(i + 1, p - 1, t)

    def evaluate(self, t: float) -> Vector3D:
        """Evaluate B-spline at parameter t"""
        if len(self.control_points) == 0:
            return Vector3D()

        result = Vector3D()
        for i in range(len(self.control_points)):
            basis = self._cox_de_boor(i, self.degree, t)
            result = result + self.control_points[i] * basis

        return result

    def tessellate(self, num_segments: int = 10) -> List[Vector3D]:
        """Tessellate curve into line segments"""
        points = []
        t_min = self.knots[self.degree]
        t_max = self.knots[-(self.degree + 1)]

        for i in range(num_segments + 1):
            t = t_min + (t_max - t_min) * i / num_segments
            points.append(self.evaluate(t))
        return points


class BezierPatch:
    """Bezier surface patch"""

    def __init__(self, control_points: List[List[Vector3D]]):
        """Control points should be a 4x4 grid for cubic patch"""
        self.control_points = control_points

    def evaluate(self, u: float, v: float) -> Vector3D:
        """Evaluate surface at parameters u,v (0-1)"""
        if not self.control_points or not self.control_points[0]:
            return Vector3D()

        # Evaluate along u direction first (get curves in v direction)
        v_curves = []
        for i in range(len(self.control_points)):
            u_points = self.control_points[i]
            if u_points:
                curve = BezierCurve(u_points)
                v_curves.append(curve.evaluate(u))

        # Then evaluate along v direction
        if v_curves:
            v_curve = BezierCurve(v_curves)
            return v_curve.evaluate(v)

        return Vector3D()

    def tessellate(self, u_segments: int = 8, v_segments: int = 8) -> Tuple[List[Vector3D], List[List[int]]]:
        """Tessellate patch into triangles"""
        vertices = []
        indices = []

        # Generate grid of vertices
        for i in range(u_segments + 1):
            for j in range(v_segments + 1):
                u = i / u_segments
                v = j / v_segments
                vertex = self.evaluate(u, v)
                vertices.append(vertex)

        # Generate triangle indices
        for i in range(u_segments):
            for j in range(v_segments):
                # Two triangles per quad
                base = i * (v_segments + 1) + j
                indices.extend([
                    base, base + 1, base + v_segments + 1,
                    base + 1, base + v_segments + 2, base + v_segments + 1
                ])

        return vertices, indices

    def normal_at(self, u: float, v: float) -> Vector3D:
        """Calculate surface normal at u,v"""
        # Use cross product of partial derivatives
        du = 0.001
        dv = 0.001

        p = self.evaluate(u, v)
        pu = self.evaluate(min(1.0, u + du), v)
        pv = self.evaluate(u, min(1.0, v + dv))

        tangent_u = pu - p
        tangent_v = pv - p

        normal = tangent_u.cross(tangent_v)
        return normal.normalize()


class Ray:
    """Ray for intersection testing"""

    def __init__(self, origin: Vector3D, direction: Vector3D):
        self.origin = origin
        self.direction = direction.normalize()

    def at(self, t: float) -> Vector3D:
        """Point along ray at distance t"""
        return self.origin + self.direction * t


class IntersectionResult:
    """Result of intersection test"""

    def __init__(self, hit: bool = False, distance: float = float('inf'),
                 point: Optional[Vector3D] = None, normal: Optional[Vector3D] = None):
        self.hit = hit
        self.distance = distance
        self.point = point
        self.normal = normal


def ray_triangle_intersect(ray: Ray, v0: Vector3D, v1: Vector3D, v2: Vector3D) -> IntersectionResult:
    """Möller-Trumbore ray-triangle intersection"""
    # Find vectors for two edges sharing v0
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    pvec = ray.direction.cross(edge2)

    # If determinant is near zero, ray lies in plane of triangle
    det = edge1.dot(pvec)
    if abs(det) < 1e-8:
        return IntersectionResult()

    inv_det = 1.0 / det

    # Calculate distance from v0 to ray origin
    tvec = ray.origin - v0

    # Calculate u parameter and test bounds
    u = tvec.dot(pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return IntersectionResult()

    # Prepare to test v parameter
    qvec = tvec.cross(edge1)

    # Calculate v parameter and test bounds
    v = ray.direction.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return IntersectionResult()

    # Calculate t, ray intersects triangle
    t = edge2.dot(qvec) * inv_det
    if t < 0:
        return IntersectionResult()

    # Calculate intersection point and normal
    point = ray.at(t)
    normal = edge1.cross(edge2).normalize()

    return IntersectionResult(True, t, point, normal)


def ray_patch_intersect(ray: Ray, patch: BezierPatch, tessellation_level: int = 4) -> IntersectionResult:
    """Ray-patch intersection using tessellation"""
    vertices, indices = patch.tessellate(tessellation_level, tessellation_level)

    closest_hit = IntersectionResult()

    # Test ray against all triangles in tessellation
    for i in range(0, len(indices), 3):
        v0 = vertices[indices[i]]
        v1 = vertices[indices[i + 1]]
        v2 = vertices[indices[i + 2]]

        hit = ray_triangle_intersect(ray, v0, v1, v2)
        if hit.hit and hit.distance < closest_hit.distance:
            closest_hit = hit

    return closest_hit


# Utility functions
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth step function"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance_point_to_line(point: Vector3D, line_start: Vector3D, line_end: Vector3D) -> float:
    """Distance from point to line segment"""
    line_vec = line_end - line_start
    point_vec = point - line_start

    line_length = line_vec.length()
    if line_length < 1e-8:
        return point_vec.length()

    t = max(0, min(1, point_vec.dot(line_vec) / (line_length * line_length)))
    projection = line_start + line_vec * t

    return (point - projection).length()


class SubdivisionSurface:
    """Base class for subdivision surface algorithms"""

    def __init__(self, control_points: List[List[Vector3D]]):
        self.control_points = control_points
        self.levels = 0

    def subdivide(self, levels: int = 1) -> 'SubdivisionSurface':
        """Apply subdivision for given number of levels"""
        surface = self
        for _ in range(levels):
            surface = surface._subdivide_once()
        surface.levels = levels
        return surface

    def _subdivide_once(self) -> 'SubdivisionSurface':
        """Override in subclasses"""
        raise NotImplementedError

    def get_mesh(self) -> Tuple[List[Vector3D], List[List[int]]]:
        """Get vertices and faces for rendering"""
        raise NotImplementedError


class CatmullClarkSurface(SubdivisionSurface):
    """Catmull-Clark subdivision surface for quadrilateral meshes"""

    def __init__(self, control_points: List[List[Vector3D]]):
        super().__init__(control_points)
        # Ensure we have a valid quad mesh
        if not self._is_valid_quad_mesh():
            raise ValueError("Catmull-Clark requires a valid quadrilateral mesh")

    def _is_valid_quad_mesh(self) -> bool:
        """Check if control points form a valid quad mesh"""
        if not self.control_points:
            return False

        rows = len(self.control_points)
        if rows < 2:
            return False

        cols = len(self.control_points[0])
        if cols < 2:
            return False

        # Check all rows have same length
        return all(len(row) == cols for row in self.control_points)

    def _subdivide_once(self) -> 'CatmullClarkSurface':
        """Apply one level of Catmull-Clark subdivision"""
        rows = len(self.control_points)
        cols = len(self.control_points[0])

        new_rows = rows * 2 - 1
        new_cols = cols * 2 - 1
        new_points = [[Vector3D() for _ in range(new_cols)] for _ in range(new_rows)]

        # Step 1: Place original points at odd positions
        for i in range(rows):
            for j in range(cols):
                new_points[i * 2][j * 2] = self.control_points[i][j]

        # Step 2: Create edge points (average of edge endpoints and adjacent face points)
        for i in range(rows):
            for j in range(cols - 1):
                # Horizontal edge points
                if i * 2 + 1 < new_rows:
                    p1 = self.control_points[i][j]
                    p2 = self.control_points[i][j + 1]
                    # For boundary edges, just average endpoints
                    # For internal edges, would need face points (simplified version)
                    edge_point = (p1 + p2) * 0.5
                    new_points[i * 2][j * 2 + 1] = edge_point

        for i in range(rows - 1):
            for j in range(cols):
                # Vertical edge points
                if j * 2 + 1 < new_cols:
                    p1 = self.control_points[i][j]
                    p2 = self.control_points[i + 1][j]
                    edge_point = (p1 + p2) * 0.5
                    new_points[i * 2 + 1][j * 2] = edge_point

        # Step 3: Create face points (average of face vertices)
        for i in range(rows - 1):
            for j in range(cols - 1):
                if i * 2 + 1 < new_rows and j * 2 + 1 < new_cols:
                    # Average of 4 corner points
                    face_point = (self.control_points[i][j] +
                                self.control_points[i][j + 1] +
                                self.control_points[i + 1][j] +
                                self.control_points[i + 1][j + 1]) * 0.25
                    new_points[i * 2 + 1][j * 2 + 1] = face_point

        # Step 4: Update original points (Catmull-Clark averaging)
        for i in range(rows):
            for j in range(cols):
                # Simplified: average with neighbors
                neighbors = []
                if i > 0: neighbors.append(self.control_points[i - 1][j])
                if i < rows - 1: neighbors.append(self.control_points[i + 1][j])
                if j > 0: neighbors.append(self.control_points[i][j - 1])
                if j < cols - 1: neighbors.append(self.control_points[i][j + 1])

                if neighbors:
                    avg_neighbor = sum(neighbors, Vector3D()) / len(neighbors)
                    # Simple averaging (full Catmull-Clark has more complex weighting)
                    new_points[i * 2][j * 2] = (self.control_points[i][j] + avg_neighbor) * 0.5

        return CatmullClarkSurface(new_points)

    def get_mesh(self) -> Tuple[List[Vector3D], List[List[int]]]:
        """Convert to renderable mesh"""
        vertices = []
        faces = []

        rows = len(self.control_points)
        cols = len(self.control_points[0]) if rows > 0 else 0

        # Create vertex list
        for i in range(rows):
            for j in range(cols):
                vertices.append(self.control_points[i][j])

        # Create quad faces
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Two triangles per quad
                base = i * cols + j
                faces.extend([
                    [base, base + 1, base + cols + 1, base + cols],  # Quad face
                ])

        return vertices, faces


class LoopSubdivision(SubdivisionSurface):
    """Loop subdivision for triangular meshes"""

    def __init__(self, vertices: List[Vector3D], faces: List[List[int]]):
        # Convert to triangle mesh representation
        self.vertices = vertices
        self.faces = faces
        self._validate_triangular_mesh()

    def _validate_triangular_mesh(self):
        """Ensure all faces are triangles"""
        for face in self.faces:
            if len(face) != 3:
                raise ValueError("Loop subdivision requires triangular faces")

    def _subdivide_once(self) -> 'LoopSubdivision':
        """Apply one level of Loop subdivision"""
        # This is a simplified implementation
        # Full Loop subdivision is more complex with edge vertex placement

        new_vertices = self.vertices.copy()
        new_faces = []

        # For each face, create 4 new triangles
        for face in self.faces:
            v0, v1, v2 = face

            # Add midpoints (simplified - should use proper Loop weights)
            m01 = (self.vertices[v0] + self.vertices[v1]) * 0.5
            m12 = (self.vertices[v1] + self.vertices[v2]) * 0.5
            m20 = (self.vertices[v2] + self.vertices[v0]) * 0.5

            m01_idx = len(new_vertices)
            new_vertices.append(m01)
            m12_idx = len(new_vertices)
            new_vertices.append(m12)
            m20_idx = len(new_vertices)
            new_vertices.append(m20)

            # Create 4 new triangles
            new_faces.extend([
                [v0, m01_idx, m20_idx],
                [m01_idx, v1, m12_idx],
                [m20_idx, m12_idx, v2],
                [m01_idx, m12_idx, m20_idx]
            ])

        return LoopSubdivision(new_vertices, new_faces)

    def get_mesh(self) -> Tuple[List[Vector3D], List[List[int]]]:
        """Get vertices and faces"""
        return self.vertices, self.faces


# Material and Texture System
class Material:
    """Material properties for rendering"""

    def __init__(self, name: str = "Default"):
        self.name = name
        self.diffuse_color = Vector3D(0.8, 0.8, 0.8)  # RGB diffuse color
        self.specular_color = Vector3D(1.0, 1.0, 1.0)  # RGB specular color
        self.shininess = 32.0  # Specular exponent
        self.transparency = 1.0  # 0 = transparent, 1 = opaque
        self.texture_path = None  # Path to texture file
        self.normal_map_path = None  # Path to normal map

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'diffuse_color': self.diffuse_color.to_list(),
            'specular_color': self.specular_color.to_list(),
            'shininess': self.shininess,
            'transparency': self.transparency,
            'texture_path': self.texture_path,
            'normal_map_path': self.normal_map_path
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Material':
        """Create from dictionary"""
        material = cls(data.get('name', 'Default'))
        material.diffuse_color = Vector3D.from_list(data.get('diffuse_color', [0.8, 0.8, 0.8]))
        material.specular_color = Vector3D.from_list(data.get('specular_color', [1.0, 1.0, 1.0]))
        material.shininess = data.get('shininess', 32.0)
        material.transparency = data.get('transparency', 1.0)
        material.texture_path = data.get('texture_path')
        material.normal_map_path = data.get('normal_map_path')
        return material


class Texture:
    """Texture class for loading and managing textures"""

    def __init__(self, path: str = None):
        self.path = path
        self.texture_id = None
        self.width = 0
        self.height = 0
        self.loaded = False

    def load(self):
        """Load texture from file (placeholder - would use PIL/Pillow in real implementation)"""
        if not self.path:
            return False

        try:
            # Placeholder for texture loading
            # In real implementation, would load image and create OpenGL texture
            self.loaded = True
            self.width = 512  # Placeholder
            self.height = 512  # Placeholder
            return True
        except Exception as e:
            print(f"Failed to load texture {self.path}: {e}")
            return False

    def bind(self):
        """Bind texture for rendering"""
        if self.loaded and self.texture_id:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)

    def unbind(self):
        """Unbind texture"""
        glBindTexture(GL_TEXTURE_2D, 0)


# Inverse Kinematics
class IKPoseSolver:
    """Inverse kinematics solver for bone chains"""

    def __init__(self, bones: List['Bone'], tolerance: float = 0.01, max_iterations: int = 100):
        self.bones = bones
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve_fabrik(self, target: Vector3D, effector_bone: 'Bone') -> bool:
        """FABRIK (Forward And Backward Reaching Inverse Kinematics) solver"""
        # Simplified FABRIK implementation
        # In practice, would need full bone hierarchy and joint constraints

        # Find bone chain from effector to root
        chain = []
        current = effector_bone
        while current:
            chain.insert(0, current)
            current = current.parent

        if len(chain) < 2:
            return False

        # Get initial positions
        positions = [Vector3D.from_tuple(bone.position) for bone in chain]
        lengths = []
        for i in range(len(positions) - 1):
            lengths.append((positions[i + 1] - positions[i]).length())

        # FABRIK algorithm (simplified)
        for iteration in range(self.max_iterations):
            # Forward pass
            positions[-1] = target

            for i in range(len(positions) - 2, -1, -1):
                r = (positions[i + 1] - positions[i]).length()
                if r > 0:
                    positions[i] = positions[i + 1] + (positions[i] - positions[i + 1]) * (lengths[i] / r)

            # Backward pass
            positions[0] = Vector3D.from_tuple(chain[0].position)  # Root stays fixed

            for i in range(len(positions) - 1):
                r = (positions[i + 1] - positions[i]).length()
                if r > 0:
                    positions[i + 1] = positions[i] + (positions[i + 1] - positions[i]) * (lengths[i] / r)

            # Check convergence
            effector_pos = positions[-1]
            if (effector_pos - target).length() < self.tolerance:
                break

        # Update bone positions
        for i, bone in enumerate(chain[1:], 1):  # Skip root
            bone.position = positions[i].to_list()

        return True

    def solve_ccd(self, target: Vector3D, effector_bone: 'Bone') -> bool:
        """Cyclic Coordinate Descent solver"""
        # Simplified CCD implementation
        for iteration in range(self.max_iterations):
            current_effector = effector_bone
            distance = (current_effector.position - target).length()

            if distance < self.tolerance:
                return True

            # Walk up the chain
            current = effector_bone
            while current.parent:
                # Rotate joint to reduce distance
                parent_pos = current.parent.position
                effector_pos = current_effector.position

                # Simple rotation towards target (simplified)
                to_effector = effector_pos - parent_pos
                to_target = target - parent_pos

                if to_effector.length() > 0 and to_target.length() > 0:
                    axis = to_effector.cross(to_target).normalize()
                    angle = to_effector.angle_to(to_target)

                    if angle > 0.01:  # Small rotation
                        # Apply rotation (simplified - would need proper bone transformation)
                        pass

                current = current.parent

        return False