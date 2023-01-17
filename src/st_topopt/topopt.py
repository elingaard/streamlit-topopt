from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.ndimage import convolve


def build_cone_filter(rmin: float, size: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a filter of the form:
    (sum_{i in N_i} w(x_i)*y)/(sum_{i in N_i} w(x_i))
    where N_i denotes the neighbourhood of element i

    Args:
    rmin: filter radius
    size(tuple): tuple with array size in x- and y direction

    Returns:
    kernel(NxN array): filter kernel where M is dependent on the filter radius
    Nsum(NxM array): sum within each neighbourhood in the filter
    """

    nely, nelx = size
    cone_kernel_1d = np.arange(-np.ceil(rmin) + 1, np.ceil(rmin))
    [dy, dx] = np.meshgrid(cone_kernel_1d, cone_kernel_1d)
    cone_kernel_2d = np.maximum(0, rmin - np.sqrt(dx**2 + dy**2))
    kernel_sums = convolve(
        np.ones((nely, nelx)), cone_kernel_2d, mode="constant", cval=0
    )
    return cone_kernel_2d, kernel_sums


class ComplianceTopOpt(ABC):
    def __init__(self, mesh, fea, volfrac, penal, max_iter, min_change, move_limit) -> None:
        self.mesh = mesh
        self.fea = fea
        self.volfrac = volfrac
        self.penal = penal
        self.max_iter = max_iter
        self.min_change = min_change
        self.move_limit = move_limit

    def elementwise_compliance(self, u):
        ce = np.sum(
            np.matmul(u[self.mesh.edof_mat], self.fea.KE) * u[self.mesh.edof_mat],
            axis=1,
        )
        ce = ce.reshape((self.mesh.nely, self.mesh.nelx), order="F")
        return ce

    def eval_compliance(self, x_phys, u):
        ce = self.elementwise_compliance(u)
        comp = np.sum(
            self.fea.Emin + x_phys**2 * (self.fea.Emax - self.fea.Emin) * ce
        )
        return comp, ce

    def eval_vol_constraint(self, x_phys):
        vol_diff = (np.sum(x_phys) / self.mesh.nele) - self.volfrac
        return vol_diff

    def eval_sensitivities(self, x_phys, ce):
        dc = (
            -self.penal
            * (self.fea.Emax - self.fea.Emin)
            * x_phys ** (self.penal - 1)
            * ce
        )
        dc -= 1e-9  # for numerical stability
        dv = np.ones((self.mesh.nely, self.mesh.nelx))
        return dc, dv

    def bisect(self, x, dc, dv):
        l1 = 1e-9
        l2 = 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            B_K = np.sqrt(-dc / dv / lmid)
            # generate new design using optimality criteria
            fixpoint = np.minimum(1.0, np.minimum(x + self.move_limit, x * B_K))
            x_new = np.maximum(0.0, np.maximum(x - self.move_limit, fixpoint))
            x_phys = self.apply_design_filter(x_new)
            vol_diff = self.eval_vol_constraint(x_phys)
            if vol_diff > 0:
                l1 = lmid
            else:
                l2 = lmid

        return x_new, x_phys

    @abstractmethod
    def apply_design_filter(self, x):
        return x

    @abstractmethod
    def apply_sensitivity_filter(self, x, dc, dv):
        return x

    @abstractmethod
    def step(self):
        pass

    def run(self):
        max_rho_change = self.min_change + 1e9
        while self.min_change < max_rho_change and self.max_iter > self.iter:
            comp, max_rho_change = self.step()

        return comp


class SensitivityFilterTopOpt(ComplianceTopOpt):
    def __init__(self, rmin, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.rho = np.ones((self.mesh.nely, self.mesh.nelx)) * self.volfrac
        self.rho_phys = self.rho.copy()
        self.filter_kernel, self.kernel_sums = build_cone_filter(
            rmin, (self.mesh.nely, self.mesh.nelx)
        )
        self.iter = 0

    def apply_design_filter(self, x):
        x_phys = x.copy()
        return x_phys

    def apply_sensitivity_filter(self, x_phys, dc, dv):
        dc = convolve(dc * x_phys, self.filter_kernel, mode="constant", cval=0)
        dc = dc / self.kernel_sums / np.maximum(1e-3, x_phys)
        return dc, dv

    def step(self):
        disp = self.fea.solve_(self.rho_phys, self.penal, sparse=True, unit_load=True)
        comp, ce = self.eval_compliance(self.rho, disp)
        dc, dv = self.eval_sensitivities(self.rho_phys, ce)
        dc, dv = self.apply_sensitivity_filter(self.rho_phys, dc, dv)
        rho_new, self.rho_phys = self.bisect(self.rho, dc, dv)

        max_rho_change = np.max(np.abs(rho_new - self.rho))
        self.rho = rho_new.copy()

        self.iter += 1

        return comp, max_rho_change


class DensityFilterTopOpt(ComplianceTopOpt):
    def __init__(self, rmin, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.rho = np.ones((self.mesh.nely, self.mesh.nelx)) * self.volfrac
        self.rho_phys = self.rho.copy()
        self.filter_kernel, self.kernel_sums = build_cone_filter(
            rmin, (self.mesh.nely, self.mesh.nelx)
        )
        self.iter = 0

    def apply_design_filter(self, x):
        x_phys = (
            convolve(x, self.filter_kernel, mode="constant", cval=0) / self.kernel_sums
        )
        return x_phys

    def apply_sensitivity_filter(self, dc, dv):
        dc = convolve(
            dc / self.kernel_sums, self.filter_kernel, mode="constant", cval=0
        )
        dv = convolve(
            dv / self.kernel_sums, self.filter_kernel, mode="constant", cval=0
        )
        return dc, dv

    def step(self):
        disp = self.fea.solve_(self.rho_phys, self.penal, sparse=True, unit_load=True)
        comp, ce = self.eval_compliance(self.rho, disp)
        dc, dv = self.eval_sensitivities(self.rho_phys, ce)
        dc, dv = self.apply_sensitivity_filter(dc, dv)
        rho_new, self.rho_phys = self.bisect(self.rho, dc, dv)

        max_rho_change = np.max(np.abs(rho_new - self.rho))
        self.rho = rho_new.copy()

        self.iter += 1

        return comp, max_rho_change


class HeavisideFilterTopOpt(ComplianceTopOpt):
    def __init__(self, rmin, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.beta = 1
        self.rho = np.ones((self.mesh.nely, self.mesh.nelx)) * self.volfrac
        self.rho_tilde = self.rho.copy()
        self.rho_phys = (
            1
            - np.exp(-self.beta * self.rho_tilde)
            + self.rho_tilde * np.exp(-self.beta)
        )
        self.filter_kernel, self.kernel_sums = build_cone_filter(
            rmin, (self.mesh.nely, self.mesh.nelx)
        )
        self.iter_beta = 0
        self.iter = 0

    def apply_design_filter(self, x):
        x_tilde = (
            convolve(x, self.filter_kernel, mode="constant", cval=0) / self.kernel_sums
        )
        self.rho_tilde = x_tilde
        x_phys = 1 - np.exp(-self.beta * x_tilde) + x_tilde * np.exp(-self.beta)
        return x_phys

    def apply_sensitivity_filter(self, x_tilde, dc, dv):
        dx = self.beta * np.exp(-self.beta * x_tilde) + np.exp(-self.beta)
        dc = convolve(
            (dc * dx) / self.kernel_sums,
            self.filter_kernel,
            mode="constant",
            cval=0,
        )
        dv = convolve(
            (dv * dx) / self.kernel_sums,
            self.filter_kernel,
            mode="constant",
            cval=0,
        )
        return dc, dv

    def step(self):
        disp = self.fea.solve_(self.rho_phys, self.penal, sparse=True, unit_load=True)
        comp, ce = self.eval_compliance(self.rho, disp)
        dc, dv = self.eval_sensitivities(self.rho_phys, ce)
        dc, dv = self.apply_sensitivity_filter(self.rho_tilde, dc, dv)
        rho_new, self.rho_phys = self.bisect(self.rho, dc, dv)

        max_rho_change = np.max(np.abs(rho_new - self.rho))
        self.rho = rho_new.copy()

        self.iter += 1
        self.iter_beta += 1

        if self.beta < 512:
            if self.iter_beta >= 50 or max_rho_change <= self.min_change:
                self.beta *= 2
                self.iter_beta = 0
                max_rho_change = self.min_change + 1e9

        return comp, max_rho_change
