import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from st_topopt.FEA import QuadMesh, LinearElasticity
from st_topopt.topopt import (
    SensitivityFilterTopOpt,
    DensityFilterTopOpt,
    HeavisideFilterTopOpt,
)


def init_MBB_beam(mesh, fea):
    # insert fixed boundary condition on left edge of the domain in x-direction
    left_wall_nodes = np.nonzero(mesh.XY[:, 0] == 0)[0]
    fea.insert_node_boundaries(node_ids=left_wall_nodes, axis=0)
    # insert vertical support in lower right corner
    fea.insert_node_boundaries(node_ids=np.array([mesh.nnodes - 1]), axis=1)
    # insert force in the middle on the right edge extending 1/10 of the domain in
    # y-direction
    fea.insert_node_load(node_id=0, load_vec=(0, -1))


def init_cantilever_beam(mesh, fea):
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


def init_bridge(mesh, fea):
    nelx, nely = (mesh.nelx, mesh.nely)
    # insert fixed boundary condition on entire lower part of the domain
    lower_wall_nodes = np.nonzero(mesh.XY[:, 1] == nely)[0]
    fea.insert_node_boundaries(node_ids=lower_wall_nodes, axis=0)
    fea.insert_node_boundaries(node_ids=lower_wall_nodes, axis=1)
    # insert distributed load across the entire upper part of the domain
    upper_wall_eles = np.arange(0, mesh.nele, step=nely)
    fea.insert_face_forces(ele_ids=upper_wall_eles, ele_face=0, load_vec=(0, -1))


def app():
    st.title("Streamlit TopOpt App")

    with st.expander("FEA parameters"):
        with st.form("fea_params"):
            c1, c2, c3 = st.columns(3)
            nelx = c1.number_input(
                "nelx",
                min_value=1,
                max_value=1000,
                value=60,
                help="Number of elements in horizontal direction",
            )
            nely = c2.number_input(
                "nely",
                min_value=1,
                max_value=1000,
                value=30,
                help="Number of elements in vertical direction",
            )
            problem_options = ["MBB", "Bridge", "Cantilever"]
            problem = c3.selectbox(
                "problem",
                options=problem_options,
                help="Choose between different standard problems",
            )

            if st.form_submit_button("Submit"):
                del st.session_state.mesh
                del st.session_state.fea
                del st.session_state.topopt

    if "mesh" or "fea" not in st.session_state:
        mesh = QuadMesh(nelx, nely)
        fea = LinearElasticity(mesh)
        if problem == "MBB":
            init_MBB_beam(mesh, fea)
        elif problem == "Bridge":
            init_bridge(mesh, fea)
        elif problem == "Cantilever":
            init_cantilever_beam()
        st.session_state["mesh"] = mesh
        st.session_state["fea"] = fea

    with st.expander("Optimization parameters"):
        with st.form("opt_params"):
            c1, c2, c3 = st.columns(3)
            volfrac = c1.number_input(
                "Volume fraction",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                help="Volume constraint defining how much of the design domain can contain material",
            )
            dens_penal = c2.number_input(
                "Intermediate density penalty",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                help="Penalty on intermediate densities which helps force the solution towards 0 and 1. Higher means stronger enforcement",
            )
            filter_radius = c3.number_input(
                "Filter radius",
                min_value=1.0,
                max_value=5.0,
                value=1.5,
                help="Filter radius. Can also be interpreted as minimum feature size.",
            )
            max_iter = c1.number_input(
                "Max iterations", min_value=1, max_value=10000, value=1000
            )
            min_change = c2.number_input(
                "Min design change",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                help="Convergence criteria based on minimum change in any design variable",
            )
            move_limit = c3.number_input(
                "Move limit",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                help="Maximum change of a design variable during one optimization step",
            )
            filter_options = ["Sensitivity", "Density", "Heaviside"]
            filter_type = c1.selectbox("Filter type", options=filter_options)

            if st.form_submit_button("Submit"):
                del st.session_state.topopt

    opt_params = {
        "mesh": st.session_state.mesh,
        "fea": st.session_state.fea,
        "volfrac": volfrac,
        "penal": dens_penal,
        "max_iter": max_iter,
        "min_change": min_change,
        "move_limit": move_limit,
        "rmin": filter_radius,
    }

    if "topopt" not in st.session_state:
        if filter_type == "Sensitivity":
            topopt = SensitivityFilterTopOpt(**opt_params)
        elif filter_type == "Density":
            topopt = DensityFilterTopOpt(**opt_params)
        elif filter_type == "Heaviside":
            topopt = HeavisideFilterTopOpt(**opt_params)

        st.session_state["topopt"] = topopt

    st.write("**Control panel**")
    c1, c2, c3, c4, _ = st.columns([0.15, 0.2, 0.15, 0.15, 0.35])
    step1_button = c1.button("Step ➡️")
    step10_button = c2.button("Step 10 ⏩")
    run_button = c3.button("Run ▶️")
    reset_button = c4.button("Reset 🔄")
    if step1_button:
        st.session_state.topopt.step()
    elif step10_button:
        for _ in range(10):
            st.session_state.topopt.step()
    elif run_button:
        st.session_state.topopt.run()
    elif reset_button:
        st.session_state.topopt = st.session_state.topopt.__class__(**opt_params)

    fig, ax = plt.subplots()
    ax.matshow(-st.session_state.topopt.rho_phys, cmap="gray", vmin=-1, vmax=0)
    ax.set_title(
        f"Comp: {st.session_state.topopt.comp :.2f}, it. {st.session_state.topopt.iter}"
    )
    st.pyplot(fig)

    with st.expander("Analysis"):
        disp = st.session_state.topopt.fea.displacement
        strain_energy_mat = st.session_state.fea.compute_strain_energy(disp)
        fig, ax = plt.subplots()
        im = ax.matshow(strain_energy_mat)
        ax.set_title("Strain energy")
        st.pyplot(fig)

        sigma_vm_mat = st.session_state.fea.compute_von_mises_stresses(disp)
        fig, ax = plt.subplots()
        im = ax.matshow(sigma_vm_mat)
        ax.set_title("Von Mises")
        st.pyplot(fig)


if __name__ == "__main__":
    app()
