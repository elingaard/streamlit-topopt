import sys
import subprocess
import time
from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

try:
    import st_topopt
except ModuleNotFoundError:
    subprocess.Popen([f"{sys.executable} setup.py install"], shell=True)
    time.sleep(60) # wait for install to finish

from st_topopt.FEA import QuadMesh, LinearElasticity
from st_topopt import benchmarks
from st_topopt.topopt import (
    SensitivityFilterTopOpt,
    DensityFilterTopOpt,
    HeavisideFilterTopOpt,
)

MAX_FIGSIZE = 16


def get_figsize_from_array(arr):
    height, width = arr.shape
    scale_factor = max(height, width) // MAX_FIGSIZE
    figsize = (width // scale_factor, height // scale_factor)
    return figsize


def matshow_to_image_buffer(mat, display_cmap=False, **kwargs):
    figsize = get_figsize_from_array(mat)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(mat, **kwargs)
    if display_cmap:
        plt.colorbar(im, ax=ax)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    return buf


def fea_parameter_selector():
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
            problem_options = ["MBB", "Bridge", "Cantilever", "Caramel"]
            problem = c3.selectbox(
                "problem",
                options=problem_options,
                help="Choose between different standard problems",
            )

            if st.form_submit_button("Submit"):
                del st.session_state.mesh
                del st.session_state.fea
                del st.session_state.topopt

    return nelx, nely, problem


def opt_parameter_selector():
    with st.expander("Optimization parameters"):
        with st.form("opt_params"):
            c1, c2, c3 = st.columns(3)
            volfrac = c1.number_input(
                "Volume fraction",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                help="""Volume constraint defining how much of the design domain can 
                contain material""",
            )
            dens_penal = c2.number_input(
                "Intermediate density penalty",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                help="""Penalty on intermediate densities which helps force the solution 
                towards 0 and 1. Higher means stronger enforcement""",
            )
            filter_radius = c3.number_input(
                "Filter radius",
                min_value=1.0,
                max_value=5.0,
                value=1.5,
                help="""Filter radius. Can also be interpreted as minimum feature size.
                """,
            )
            max_iter = c1.number_input(
                "Max iterations", min_value=1, max_value=10000, value=1000
            )
            min_change = c2.number_input(
                "Min design change",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                help="""Convergence criteria based on minimum change in any design 
                variable""",
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

    return opt_params, filter_type


def design_plot(container):
    with container.container():
        st.text(
            f"Comp: {st.session_state.topopt.comp :.2f}, it. {st.session_state.topopt.iter}"
        )
        rho_phys = -st.session_state.topopt.rho_phys
        img_buf = matshow_to_image_buffer(
            rho_phys, display_cmap=False, vmin=-1, vmax=0, cmap="gray"
        )
        st.image(img_buf)


def opt_control_panel(opt_params, plot_container):
    st.write("**Optimization control panel**")
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
        topopt = st.session_state.topopt
        topopt.step()
        upd_time = 2.0
        start_time = time.time()
        st.session_state.running_opt = True
        while (
            topopt.max_rho_change > topopt.min_change and topopt.iter < topopt.max_iter
        ):
            topopt.step()
            end_time = time.time()
            if (end_time - start_time) > upd_time:
                design_plot(plot_container)
                start_time = time.time()
        else:
            st.success("Optimization finished!")
            st.balloons()
            time.sleep(2)
            st.experimental_rerun()
    elif reset_button:
        st.session_state.topopt = st.session_state.topopt.__class__(**opt_params)


def app():
    st.title("Streamlit TopOpt App")

    st.session_state.running_opt = False

    nelx, nely, problem = fea_parameter_selector()

    if "mesh" or "fea" not in st.session_state:
        mesh = QuadMesh(nelx, nely)
        fea = LinearElasticity(mesh)
        if problem == "MBB":
            benchmarks.init_MBB_beam(mesh, fea)
        elif problem == "Bridge":
            benchmarks.init_bridge(mesh, fea)
        elif problem == "Cantilever":
            benchmarks.init_cantilever_beam()
        elif problem == "Caramel":
            benchmarks.init_caramel(mesh, fea)
        st.session_state["mesh"] = mesh
        st.session_state["fea"] = fea

    opt_params, filter_type = opt_parameter_selector()

    if "topopt" not in st.session_state:
        if filter_type == "Sensitivity":
            topopt = SensitivityFilterTopOpt(**opt_params)
        elif filter_type == "Density":
            topopt = DensityFilterTopOpt(**opt_params)
        elif filter_type == "Heaviside":
            topopt = HeavisideFilterTopOpt(**opt_params)

        st.session_state["topopt"] = topopt

    if "img_container" not in st.session_state:
        st.session_state["img_container"] = st.container()

    opt_container = st.empty()
    opt_control_panel(opt_params, opt_container)
    if st.session_state.running_opt is False:
        design_plot(st.container())

    with st.expander("Analysis"):
        plot_type = st.radio(
            "What do you want to plot?",
            options=["Von Mises", "Strain energy", "Log metrics"],
            horizontal=True,
        )
        if plot_type == "Strain energy":
            disp = st.session_state.topopt.fea.displacement
            strain_energy_mat = st.session_state.fea.compute_strain_energy(disp)
            img_buf = matshow_to_image_buffer(strain_energy_mat, display_cmap=True)
            st.image(img_buf, caption=plot_type)
        elif plot_type == "Von Mises":
            disp = st.session_state.topopt.fea.displacement
            sigma_vm_mat = st.session_state.fea.compute_von_mises_stresses(disp)
            img_buf = matshow_to_image_buffer(sigma_vm_mat, display_cmap=True)
            st.image(img_buf, caption=plot_type)
        elif plot_type == "Log metrics":
            logger = st.session_state.topopt.logger
            if logger.metrics:
                metric_options = list(logger.metrics.keys())
                metric_options.remove("iter")
                metric_key = st.selectbox("Metric", options=metric_options)
                fig, ax = plt.subplots()
                ax.plot(logger.metrics["iter"], logger.metrics[metric_key], "-o")
                ax.set_xlabel("Iter.")
                ax.set_ylabel(metric_key)
                st.pyplot(fig)
            else:
                st.info("No metrics logged yet")


if __name__ == "__main__":
    app()
