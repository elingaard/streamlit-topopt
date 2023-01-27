import numpy as np

from st_topopt.FEA import QuadMesh2D, LinearElasticity, HeatConduction


def init_MBB_beam(mesh: QuadMesh2D, fea: LinearElasticity):
    # insert fixed boundary condition on left edge of the domain in x-direction
    left_wall_nodes = np.nonzero(mesh.XY[:, 0] == 0)[0].astype(int)
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=0)
    # insert vertical support in lower right corner
    fea.insert_node_boundaries(node_ids=np.array([mesh.nnodes - 1]), axis=1)
    # insert force in upper left corner
    left_corner_node = 0
    fea.insert_node_loads(node_ids=np.array([left_corner_node]), load_mag=(0.0, -1.0))


def init_cantilever_beam(mesh: QuadMesh2D, fea: LinearElasticity):
    nelx, nely = (mesh.nelx, mesh.nely)
    # insert fixed boundary condition on left edge of the domain in both directions
    left_wall_nodes = np.nonzero(mesh.XY[:, 0] == 0)[0].astype(int)
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=0)
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=1)
    # insert element face force in the middle on the right edge of the domain in
    # y-direction
    load_width = 1 / 20
    load_ele_width = int(np.ceil(load_width * nely)) // 2
    mid_ele_right_wall = (nelx - 1) * nely + nely // 2
    load_eles = np.arange(
        mid_ele_right_wall - load_ele_width,
        mid_ele_right_wall + load_ele_width + 1,
        dtype=int,
    )
    fea.insert_face_forces(ele_ids=load_eles, ele_face=1, load_mag=(0.0, -1.0))


def init_bridge(mesh: QuadMesh2D, fea: LinearElasticity):
    # insert fixed boundary condition on entire lower part of the domain
    lower_wall_nodes = np.nonzero(mesh.XY[:, 1] == mesh.nely)[0]
    fea.insert_node_boundaries(node_ids=lower_wall_nodes, axis=0)
    fea.insert_node_boundaries(node_ids=lower_wall_nodes, axis=1)
    # insert distributed load across the entire upper part of the domain
    upper_wall_eles = np.arange(0, mesh.nele, step=mesh.nely, dtype=int)
    fea.insert_face_forces(ele_ids=upper_wall_eles, ele_face=0, load_mag=(0.0, -1.0))


def init_caramel(mesh: QuadMesh2D, fea: LinearElasticity):
    nelx, nely = (mesh.nelx, mesh.nely)
    # insert fixed boundary condition on left and right wall
    left_wall_nodes = np.nonzero(mesh.XY[:, 0] == 0)[0]
    right_wall_nodes = np.nonzero(mesh.XY[:, 0] == nelx)[0]
    constrained_nodes = np.stack((left_wall_nodes, right_wall_nodes)).flatten()
    fea.insert_node_boundaries(node_ids=constrained_nodes, axis=0)
    fea.insert_node_boundaries(node_ids=constrained_nodes, axis=1)
    # insert element force load in the middle of the domain
    mid_ele = (nelx * nely) // 2 - nely // 2
    load_width = 1 / 20
    load_ele_width = int(np.ceil(load_width * nely)) // 2
    idx_range = np.arange(-load_ele_width, load_ele_width + 1, dtype=int)
    I, J = np.meshgrid(idx_range, idx_range)
    load_eles = (mid_ele + I + J * nely).flatten()
    fea.insert_element_forces(ele_ids=load_eles, load_mag=(0.0, -1.0))


def init_heat_sink(mesh: QuadMesh2D, fea: HeatConduction):
    # temperature=0 on a small area on the middle of the left wall
    bound_width = 1 / 20
    bound_ele_width = int(np.ceil(bound_width * mesh.nely)) // 2
    mid_ele_left_wall = mesh.nely // 2
    bound_eles = np.arange(
        mid_ele_left_wall - bound_ele_width,
        mid_ele_left_wall + bound_ele_width + 1,
        dtype=int,
    )
    fea.insert_node_boundaries(node_ids=bound_eles, axis=0)
    # constant heating across entire domain
    fea.insert_node_loads(node_ids=np.arange(mesh.ndof), load_mag=(0.01,))
