import numpy as np

from st_topopt.FEA import QuadMesh, LinearElasticity


def init_MBB_beam(mesh: QuadMesh, fea: LinearElasticity):
    # insert fixed boundary condition on left edge of the domain in x-direction
    left_wall_nodes = np.nonzero(mesh.XY[:, 0] == 0)[0]
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=0)
    # insert vertical support in lower right corner
    fea.insert_node_boundaries(node_ids=np.array([mesh.nnodes - 1]), axis=1)
    # insert force in upper left corner
    fea.insert_node_load(node_id=0, load_vec=(0, -1))


def init_cantilever_beam(mesh: QuadMesh, fea: LinearElasticity):
    nelx, nely = (mesh.nelx, mesh.nely)
    # insert fixed boundary condition on left edge of the domain in both directions
    left_wall_nodes = np.nonzero(mesh.XY[:, 0] == 0)[0]
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=0)
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=1)
    # insert force in the middle on the right edge extending 1/10 of the domain in
    # y-direction
    load_extent_y = 1 / 10
    ele_extent = nely * load_extent_y / 2
    mid_ele_right_wall = (nelx - 1) * nely + nely // 2
    load_start_ele = int(np.floor(mid_ele_right_wall - ele_extent))
    load_end_ele = int(np.ceil(mid_ele_right_wall + ele_extent))
    right_wall_load_ele = np.arange(load_start_ele, load_end_ele)
    fea.insert_face_forces(ele_ids=right_wall_load_ele, ele_face=1, load_vec=(0, -1))


def init_bridge(mesh: QuadMesh, fea: LinearElasticity):
    nelx, nely = (mesh.nelx, mesh.nely)
    # insert fixed boundary condition on entire lower part of the domain
    lower_wall_nodes = np.nonzero(mesh.XY[:, 1] == nely)[0]
    fea.insert_node_boundaries(node_ids=lower_wall_nodes, axis=0)
    fea.insert_node_boundaries(node_ids=lower_wall_nodes, axis=1)
    # insert distributed load across the entire upper part of the domain
    upper_wall_eles = np.arange(0, mesh.nele, step=nely)
    fea.insert_face_forces(ele_ids=upper_wall_eles, ele_face=0, load_vec=(0, -1))
