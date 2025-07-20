from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def plot_convex_hull(points, **polygon_kwargs):
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    closed = polygon_kwargs.get("closed", True)
    edgecolor = polygon_kwargs.get("edgecolor", True)
    facecolor = polygon_kwargs.get("facecolor", True)
    linewidth = polygon_kwargs.get("linewidth", True)
    polygon = Polygon(hull_vertices, closed=True, edgecolor='red', facecolor='none', linewidth=1)
    return polygon
