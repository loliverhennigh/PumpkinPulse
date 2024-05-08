from __future__ import annotations
from tqdm import tqdm
from multiprocessing import Pool
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Pnt
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools


def load_brep(filename: str) -> TopoDS_Shape:
    shape = TopoDS_Shape()
    builder = BRep_Builder()
    BRepTools.Read_s(shape, filename, builder)
    return shape


def process_batch(
    brep_model: TopoDS_Shape, batch: list[tuple[float, float, float]]
) -> list[tuple[tuple[float, float, float], int]]:
    results = []
    for point in tqdm(batch):
        classifier = BRepClass3d_SolidClassifier()
        classifier.Load(brep_model)
        classifier.Perform(gp_Pnt(*point), 1e-6)
        state = classifier.State()
        results.append((point, state))
    return results


def main():
    brep_model = load_brep("reactor.brep")
    nr_points = 1000000
    point_cloud = nr_points * [(0, 0, 0)]
    results = process_batch(brep_model, point_cloud)


if __name__ == "__main__":
    main()
