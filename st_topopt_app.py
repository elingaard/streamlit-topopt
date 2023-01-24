import sys
import site
from importlib import reload
import subprocess
import time
from io import BytesIO

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

with st.spinner("Installing packages..."):
    try:
        import st_topopt
    except ModuleNotFoundError:
        subprocess.call([f"{sys.executable} setup.py install"], shell=True)
    finally:
        reload(site)  # force reload of sys.path
        from st_topopt.FEA import QuadMesh, LinearElasticity
        from st_topopt import benchmarks
        from st_topopt.utils import PillowGIFWriter, matshow_to_image_buffer
        from st_topopt.topopt import (
            SensitivityFilterTopOpt,
            DensityFilterTopOpt,
            HeavisideFilterTopOpt,
        )


def fea_parameter_selector():
    with st.form("fea_params"):
        c1, c2, c3, c4 = st.columns(4)
        nelx = c1.number_input(
            "Nr. elements in x",
            min_value=1,
            max_value=1000,
            value=80,
            help="Number of elements in horizontal direction",
        )
        nely = c2.number_input(
            "Nr. elements in y",
            min_value=1,
            max_value=1000,
            value=40,
            help="Number of elements in vertical direction",
        )
        solver_disp_names = {"sparse-direct": "direct", "mgcg": "iterative"}
        solver = c3.selectbox(
            "Solver",
            options=["sparse-direct", "mgcg"],
            format_func=lambda k: solver_disp_names[k],
            help="""It is recommened to use the direct solver for smaller problems and 
            the iterative solver for larger problems""",
        )
        problem_options = ["MBB", "Bridge", "Cantilever", "Caramel"]
        problem = c4.selectbox(
            "Problem",
            options=problem_options,
            help="Choose between the different standard problems visualized above",
        )

        if st.form_submit_button("Submit"):
            try:
                del st.session_state.mesh
                del st.session_state.fea
                del st.session_state.topopt
            except AttributeError:
                pass

    return nelx, nely, solver, problem


def opt_parameter_selector():
    with st.form("opt_params"):
        c1, c2, c3 = st.columns(3)
        volfrac = c1.number_input(
            "Volume fraction",
            min_value=0.05,
            max_value=1.0,
            value=0.3,
            help="""Volume constraint defining how much of the design domain can contain 
            material""",
        )
        dens_penal = c2.number_input(
            "Intermediate density penalty",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            help="""Penalty on intermediate densities which helps force the solution 
            towards a binary design. Higher means stronger enforcement.""",
        )
        filter_radius = c3.number_input(
            "Filter radius",
            min_value=1.0,
            max_value=100.0,
            value=1.5,
            help="""Filter radius in elements. Can also be interpreted as minimum 
            feature size.""",
        )
        max_iter = c1.number_input(
            "Max iterations",
            min_value=1,
            max_value=10000,
            value=100,
            help="""Maximum number of iterations the optimization is allowed to run.""",
        )
        min_change = c2.number_input(
            "Min design change",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            help="""The optimization will change when the maximum change in any design
            variable is less than this value.""",
        )
        move_limit = c3.number_input(
            "Move limit",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="""Upper limit on how much a single design variable can change during 
            one optimization step""",
        )
        filter_options = ["Sensitivity", "Density", "Heaviside"]
        filter_type = c1.selectbox(
            "Filter type",
            options=filter_options,
            help="""Sensitive filter: apply filter to the sensitivities of the 
        objective function. Density filter: apply filter directly to the densities.
        Heaviside filter: gradually apply an approximation of the Heaviside
        function to the densities to strongly enforce binary designs.""",
        )

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


def show_opt_status(placeholder: st.container):
    """Show results after one optimization step in terms of compliance, iteration,
    and design field"""
    with placeholder.container():
        iter = st.session_state.topopt.iter
        comp = st.session_state.topopt.comp
        c1, c2, _ = st.columns([0.25, 0.25, 0.5])
        c1.metric("Compliance", value=np.round(comp, 2))
        c2.metric("Iteration", value=iter)
        rho_phys = st.session_state.topopt.rho_phys
        img_buf = matshow_to_image_buffer(
            -rho_phys, display_cmap=False, vmin=-1, vmax=0, cmap="gray"
        )
        st.image(img_buf)


def run_optimization():
    stop_placeholder = st.empty()
    design_placeholder = st.empty()
    topopt = st.session_state.topopt
    topopt.max_rho_change = 1e6
    upd_time = 2.0
    start_time = time.time()
    while topopt.max_rho_change > topopt.min_change and topopt.iter < topopt.max_iter:
        if stop_placeholder.button("Stop optimization ⛔", key=topopt.iter):
            break
        topopt.step()
        end_time = time.time()
        if (end_time - start_time) > upd_time:
            show_opt_status(design_placeholder)
            start_time = time.time()
        if st.session_state.record:
            st.session_state.gif_writer.add_frame(topopt.rho_phys)
    else:
        st.success("Optimization finished!")
        st.balloons()
        time.sleep(2)
        st.experimental_rerun()


def opt_control_panel(opt_params):
    c1, c2, c3, c4, c5 = st.columns(5)
    step1_button = c1.button("Step ➡️")
    step10_button = c2.button("Step 10 ⏩")
    run_button = c3.button("Run ▶️")
    reset_button = c4.button("Reset 🔄")
    record_button = c5.checkbox(
        "Record 🎥",
        help="""Enable/disable recording of optimization process. If enabled, the video 
        can be shown and downloaded in the analysis section.""",
    )
    if step1_button:
        st.session_state.topopt.step()
    elif step10_button:
        for _ in range(10):
            st.session_state.topopt.step()
    elif run_button:
        st.session_state.gif_writer.reset_()
        with st.spinner("Running optimization..."):
            run_optimization()
    elif reset_button:
        st.session_state.topopt = st.session_state.topopt.__class__(**opt_params)
    elif record_button:
        st.session_state.record = True


def download_design():
    with BytesIO() as buffer:
        np.save(buffer, st.session_state.topopt.rho_phys)
        st.download_button(
            label="Download design ⬇️",
            data=buffer,
            file_name="design.npy",
            help="Download the design as a numpy array (.npy)",
        )


def app():
    st.title("Compliance topology optimizer")

    intro_text = """Welcome! 👋 this is an app for quickly trying out different solvers,
    hyper-parameters, and filters on a set of standard problems for compliance 
    topology optimization."""
    st.write(intro_text)

    topopt_text = """*The goal of topology optimization is to optimize the material
    layout within a given design space, subject to a set of constraints and loads. 
    In this app only the compliance objective with a volume constraint is considered.
    In layman terms this can be described as maximimizing the stiffness of the design
    for a given amount of material. The design variables are the densities in each 
    element of the discretized design domain (mesh).* """

    with st.expander("What is compliance topology optimization?"):
        st.markdown(topopt_text)

    intro_text2 = """Choose the optimization problem and parameters in the the two 
    expanders below, and use the control panel to perform a single step or run the 
    entire optimization. """
    st.write(intro_text2)

    if "gif_writer" not in st.session_state:
        st.session_state.record = False
        st.session_state["gif_writer"] = PillowGIFWriter()

    with st.expander("Problem setup and solver"):
        st.image(
            "imgs/TO_problems.png",
            use_column_width=True,
            caption="""Design domain, loads and boundary conditions for each of the 
            standard problems""",
        )
        nelx, nely, solver, problem = fea_parameter_selector()

    if "mesh" not in st.session_state or "fea" not in st.session_state:
        mesh = QuadMesh(nelx, nely)
        fea = LinearElasticity(mesh, solver=solver)
        if problem == "MBB":
            benchmarks.init_MBB_beam(mesh, fea)
        elif problem == "Bridge":
            benchmarks.init_bridge(mesh, fea)
        elif problem == "Cantilever":
            benchmarks.init_cantilever_beam(mesh, fea)
        elif problem == "Caramel":
            benchmarks.init_caramel(mesh, fea)
        st.session_state["mesh"] = mesh
        st.session_state["fea"] = fea
        st.session_state.gif_writer.reset_()

    if (
        st.session_state.mesh.ndof > 5e4
        and st.session_state.fea.solver == "sparse-direct"
    ):
        st.warning(
            """You are using the direct solver for a system with more than 50k 
        degrees-of-freedom. For better performance use the iterative solver. """
        )

    with st.expander("Optimization parameters"):
        opt_params, filter_type = opt_parameter_selector()

    if "topopt" not in st.session_state:
        if filter_type == "Sensitivity":
            topopt = SensitivityFilterTopOpt(**opt_params)
        elif filter_type == "Density":
            topopt = DensityFilterTopOpt(**opt_params)
        elif filter_type == "Heaviside":
            topopt = HeavisideFilterTopOpt(**opt_params)

        st.session_state["topopt"] = topopt

    st.subheader("Optimization control panel")
    opt_control_panel(opt_params)
    show_opt_status(st.container())
    download_design()

    with st.expander("Analysis"):
        analysis_type = st.radio(
            "What do you want to show?",
            options=["Von Mises", "Strain energy", "Log metrics", "GIF"],
            index=0,
            horizontal=True,
        )

        if analysis_type == "Strain energy" or analysis_type == "Von Mises":
            disp = st.session_state.topopt.fea.displacement
            dens_mask = st.session_state.topopt.rho_phys < 0.1
            if analysis_type == "Strain energy":
                field_mat = st.session_state.fea.compute_strain_energy(disp)
            elif analysis_type == "Von Mises":
                field_mat = st.session_state.fea.compute_von_mises_stresses(disp)
            field_mat[dens_mask] = np.nan
            img_buf = matshow_to_image_buffer(field_mat, display_cmap=True)
            st.image(img_buf, caption=analysis_type)

        elif analysis_type == "Log metrics":
            logger = st.session_state.topopt.logger
            if logger.metrics:
                log_df = pd.DataFrame(logger.metrics)
                metric_options = log_df.columns.tolist()
                metric_options.remove("iter")
                metric_key = st.selectbox("Metric", options=metric_options)
                log_chart = (
                    alt.Chart(log_df)
                    .mark_line(point=True)
                    .encode(
                        x="iter:Q", y=f"{metric_key}:Q", tooltip=["iter", metric_key]
                    )
                    .interactive()
                )
                st.altair_chart(log_chart, use_container_width=True)
            else:
                st.info("No metrics logged yet")

        elif analysis_type == "GIF":
            if st.session_state.gif_writer.frames:
                with BytesIO() as gif_buffer:
                    st.session_state.gif_writer.save_bytes_(gif_buffer)
                    st.image(gif_buffer)
                    st.download_button(
                        label="Download GIF ⬇️",
                        data=gif_buffer,
                        file_name="opt.gif",
                        help="Download the optimization video as a GIF",
                    )
            else:
                st.info("No video recorded yet")


if __name__ == "__main__":
    app()
