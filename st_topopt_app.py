import sys
import subprocess
import time
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import altair as alt
import streamlit as st

try:
    import st_topopt
except ModuleNotFoundError:
    subprocess.Popen([f"{sys.executable} setup.py install"], shell=True)
    time.sleep(60)  # wait for install to finish

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


class PillowGIFRecorder:
    def __init__(self) -> None:
        self.frames = []
        self.max_size = 800

    @staticmethod
    def to_grayscale_img(rho):
        rho_img = ((1 - rho) * 255).astype(np.uint8)
        return rho_img

    def resize_(self, img):
        H, W = img.size
        max_dim = max(H, W)
        scale_factor = self.max_size / max_dim
        return img.resize((int(H * scale_factor), int(W * scale_factor)), Image.NEAREST)

    def add_frame(self, rho):
        assert rho.ndim == 2

        rho_img = self.to_grayscale_img(rho)
        pil_img = Image.fromarray(rho_img, "L").convert("P")
        pil_img = self.resize_(pil_img)

        pil_img = ImageOps.expand(pil_img, border=20, fill=255)
        draw = ImageDraw.Draw(pil_img)
        draw.text((20, 0), f"It. {len(self.frames)}")

        self.frames.append(pil_img)

    def reset_(self):
        self.frames = []

    def save_bytes_(self):
        gif_bytes = BytesIO()
        self.frames[0].save(
            gif_bytes,
            format="gif",
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=1000,
            loop=0,
        )
        return gif_bytes


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
            c1, c2, c3, c4 = st.columns(4)
            nelx = c1.number_input(
                "nelx",
                min_value=1,
                max_value=1000,
                value=120,
                help="Number of elements in horizontal direction",
            )
            nely = c2.number_input(
                "nely",
                min_value=1,
                max_value=1000,
                value=60,
                help="Number of elements in vertical direction",
            )
            solver_disp_names = {"sparse-direct": "direct", "mgcg": "iterative"}
            solver = c3.selectbox(
                "Solver",
                options=["sparse-direct", "mgcg"],
                format_func=lambda k: solver_disp_names[k],
                help="""Use the direct solver for smaller problems and the iterative 
                solver for larger problems """,
            )
            problem_options = ["MBB", "Bridge", "Cantilever", "Caramel"]
            problem = c4.selectbox(
                "problem",
                options=problem_options,
                help="Choose between different standard problems",
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
                max_value=100.0,
                value=1.5,
                help="""Filter radius. Can also be interpreted as minimum feature size.
                """,
            )
            max_iter = c1.number_input(
                "Max iterations", min_value=1, max_value=10000, value=100
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


def show_design_field(placeholder: st.container):
    with placeholder.container():
        iter = st.session_state.topopt.iter
        comp = st.session_state.topopt.comp
        st.text(f"Comp: {comp :.2f}, it. {iter}")
        rho_phys = st.session_state.topopt.rho_phys
        img_buf = matshow_to_image_buffer(
            -rho_phys, display_cmap=False, vmin=-1, vmax=0, cmap="gray"
        )
        st.image(img_buf)


def run_optimization():
    placeholder = st.empty()
    topopt = st.session_state.topopt
    topopt.step()
    upd_time = 2.0
    start_time = time.time()
    while topopt.max_rho_change > topopt.min_change and topopt.iter < topopt.max_iter:
        topopt.step()
        end_time = time.time()
        if (end_time - start_time) > upd_time:
            show_design_field(placeholder)
            start_time = time.time()
        if st.session_state.record:
            st.session_state.vid_recorder.add_frame(topopt.rho_phys)
    else:
        st.success("Optimization finished!")
        st.balloons()
        time.sleep(2)
        st.experimental_rerun()


def opt_control_panel(opt_params):
    st.subheader("Optimization control panel")
    c1, c2, c3, c4, c5 = st.columns(5)
    step1_button = c1.button("Step ➡️")
    step10_button = c2.button("Step 10 ⏩")
    run_button = c3.button("Run ▶️")
    reset_button = c4.button("Reset 🔄")
    record_button = c5.checkbox("Record 🎥")
    if step1_button:
        st.session_state.topopt.step()
    elif step10_button:
        for _ in range(10):
            st.session_state.topopt.step()
    elif run_button:
        st.session_state.vid_recorder.reset_()
        with st.spinner("Running optimization..."):
            run_optimization()
    elif reset_button:
        st.session_state.topopt = st.session_state.topopt.__class__(**opt_params)
    elif record_button:
        st.session_state.record = True


def download_design():
    with BytesIO() as buffer:
        # Write array to buffer
        np.save(buffer, st.session_state.topopt.rho_phys)
        st.download_button(
            label="Download design ⬇️",
            data=buffer,
            file_name="design.npy",
            help="Download the design as a numpy array (.npy)",
        )


def app():
    st.title("Topology optimizer")

    intro_text = """Welcome! 👋 this is an app for quickly trying out different solvers,
    hyperparameters, and filters on a few different benchmark topology optimzation 
    problems."""
    st.write(intro_text)
    intro_text2 = """Choose the optimization problem and parameters in the the two 
    expanders below, and use the control panel to perform a single step or run the 
    entire optimization."""
    st.write(intro_text2)

    if "vid_recorder" not in st.session_state:
        st.session_state.record = False
        st.session_state["vid_recorder"] = PillowGIFRecorder()

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
        st.session_state.vid_recorder.reset_()

    opt_params, filter_type = opt_parameter_selector()

    if "topopt" not in st.session_state:
        if filter_type == "Sensitivity":
            topopt = SensitivityFilterTopOpt(**opt_params)
        elif filter_type == "Density":
            topopt = DensityFilterTopOpt(**opt_params)
        elif filter_type == "Heaviside":
            topopt = HeavisideFilterTopOpt(**opt_params)

        st.session_state["topopt"] = topopt

    opt_control_panel(opt_params)
    show_design_field(st.container())
    download_design()

    with st.expander("Analysis"):
        plot_var = st.radio(
            "What do you want to plot?",
            options=["Von Mises", "Strain energy", "Log metrics", "Video"],
            horizontal=True,
        )

        if plot_var == "Strain energy" or plot_var == "Von Mises":
            disp = st.session_state.topopt.fea.displacement
            dens_mask = st.session_state.topopt.rho_phys < 0.1
            if plot_var == "Strain energy":
                var_mat = st.session_state.fea.compute_strain_energy(disp)
            elif plot_var == "Von Mises":
                var_mat = st.session_state.fea.compute_von_mises_stresses(disp)
            var_mat[dens_mask] = np.nan
            img_buf = matshow_to_image_buffer(var_mat, display_cmap=True)
            st.image(img_buf, caption=plot_var)

        elif plot_var == "Log metrics":
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

        elif plot_var == "Video":
            if st.session_state.vid_recorder.frames:
                gif_bytes = st.session_state.vid_recorder.save_bytes_()
                st.image(gif_bytes)
                st.download_button(
                    label="Download video ⬇️",
                    data=gif_bytes,
                    file_name="opt.gif",
                    help="Download the optimization video",
                )
            else:
                st.info("No video recorded yet")


if __name__ == "__main__":
    app()
