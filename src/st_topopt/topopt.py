from abc import ABC, abstractmethod
from typing import Tuple, Any
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve


def build_cone_filter(
    rmin: float, size: Tuple[int, int]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build a 2D cone filter kernel and the corresponding kernel sums"""
    nely, nelx = size
    cone_kernel_1d = np.arange(-np.ceil(rmin) + 1, np.ceil(rmin))
    [dy, dx] = np.meshgrid(cone_kernel_1d, cone_kernel_1d)
    cone_kernel_2d = np.maximum(0, rmin - np.sqrt(dx**2 + dy**2))
    kernel_sums = convolve(
        np.ones((nely, nelx)), cone_kernel_2d, mode="constant", cval=0
    )
    return cone_kernel_2d, kernel_sums


class MetricLogger:
    def __init__(self) -> None:
        self.metrics: dict = defaultdict(list)

    def log_metric(self, key: str, value):
        self.metrics[key].append(value)


class ComplianceTopOpt(ABC):
    def __init__(
        self,
        mesh: Any,
        fea: Any,
        volfrac: float,
        penal: float,
        max_iter: int,
        min_change: float,
        move_limit: float,
    ) -> None:
        self.mesh = mesh
        self.fea = fea
        self.logger = MetricLogger()
        self.volfrac = volfrac
        self.penal = penal
        self.max_iter = max_iter
        self.min_change = min_change
        self.move_limit = move_limit
        self.rho = np.ones((self.mesh.nely, self.mesh.nelx)) * self.volfrac
        self.rho_phys = self.rho.copy()
        disp = self.fea.solve_(self.rho_phys, self.penal, unit_load=True)
        self.comp = self.global_compliance(self.elementwise_compliance(disp))
        self.iter = 0

    @abstractmethod
    def design_filter(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def sensitivity_filter(
        self, dc: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    def elementwise_compliance(
        self, u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ce = np.sum(
            np.matmul(u[self.mesh.edof_mat], self.fea.KE) * u[self.mesh.edof_mat],
            axis=1,
        )
        ce = ce.reshape((self.mesh.nely, self.mesh.nelx), order="F")
        return ce

    def global_compliance(self, ce: npt.NDArray[np.float64]) -> float:
        comp = np.sum(
            self.fea.Emin + self.rho_phys**2 * (self.fea.Emax - self.fea.Emin) * ce
        )
        return comp

    def eval_vol_constraint(self) -> float:
        vol_diff = (np.sum(self.rho_phys) / self.mesh.nele) - self.volfrac
        return vol_diff

    def calc_sensitivities(
        self, ce: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dc = (
            -self.penal
            * (self.fea.Emax - self.fea.Emin)
            * self.rho_phys ** (self.penal - 1)
            * ce
        )
        dc -= 1e-9  # for numerical stability
        dv = np.ones((self.mesh.nely, self.mesh.nelx))
        return dc, dv

    def bisect(
        self, dc: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        l1 = 1e-9
        l2 = 1e9
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            B_K = np.sqrt(-dc / dv / lmid)
            # generate new design using optimality criteria
            fixpoint = np.minimum(
                1.0, np.minimum(self.rho + self.move_limit, self.rho * B_K)
            )
            rho_new = np.maximum(0.0, np.maximum(self.rho - self.move_limit, fixpoint))
            self.rho_phys = self.design_filter(rho_new)
            self.vol_diff = self.eval_vol_constraint()
            if self.vol_diff > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new

    def step(self) -> None:
        disp = self.fea.solve_(self.rho_phys, self.penal, unit_load=True)
        ce = self.elementwise_compliance(disp)
        comp = self.global_compliance(ce)
        dc, dv = self.calc_sensitivities(ce)
        dc, dv = self.sensitivity_filter(dc, dv)
        rho_new = self.bisect(dc, dv)

        self.max_rho_change = np.max(np.abs(rho_new - self.rho))
        self.rho = rho_new.copy()
        self.comp = comp
        self.iter += 1

        self.logger.log_metric("iter", self.iter)
        self.logger.log_metric("compliance", comp)
        self.logger.log_metric("max_rho_change", self.max_rho_change)
        self.logger.log_metric("vol_diff", self.vol_diff)

    def run(self) -> None:
        self.step()
        while self.max_rho_change > self.min_change and self.iter < self.max_iter:
            self.step()


class SensitivityFilterTopOpt(ComplianceTopOpt):
    def __init__(self, rmin: float, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.filter_kernel, self.kernel_sums = build_cone_filter(
            rmin, (self.mesh.nely, self.mesh.nelx)
        )

    def design_filter(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x_phys = x.copy()
        return x_phys

    def sensitivity_filter(
        self, dc: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dc = convolve(dc * self.rho_phys, self.filter_kernel, mode="constant", cval=0)
        dc = dc / self.kernel_sums / np.maximum(1e-3, self.rho_phys)
        return dc, dv


class DensityFilterTopOpt(ComplianceTopOpt):
    def __init__(self, rmin: float, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.filter_kernel, self.kernel_sums = build_cone_filter(
            rmin, (self.mesh.nely, self.mesh.nelx)
        )

    def design_filter(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x_phys = (
            convolve(x, self.filter_kernel, mode="constant", cval=0) / self.kernel_sums
        )
        return x_phys

    def sensitivity_filter(
        self, dc: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dc = convolve(
            dc / self.kernel_sums, self.filter_kernel, mode="constant", cval=0
        )
        dv = convolve(
            dv / self.kernel_sums, self.filter_kernel, mode="constant", cval=0
        )
        return dc, dv


class HeavisideFilterTopOpt(ComplianceTopOpt):
    def __init__(self, rmin: float, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.beta = 1
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

    def design_filter(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.rho_tilde = (
            convolve(x, self.filter_kernel, mode="constant", cval=0) / self.kernel_sums
        )
        x_phys = (
            1
            - np.exp(-self.beta * self.rho_tilde)
            + self.rho_tilde * np.exp(-self.beta)
        )
        return x_phys

    def sensitivity_filter(
        self, dc: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dx = self.beta * np.exp(-self.beta * self.rho_tilde) + np.exp(-self.beta)
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

    def step(self) -> None:
        super().step()
        self.iter_beta += 1

        if self.beta < 512:
            if self.iter_beta >= 50 or self.max_rho_change <= self.min_change:
                self.beta *= 2
                self.iter_beta = 0
                self.max_rho_change = self.min_change + 1e9

        self.logger.log_metric("iter_beta", self.iter_beta)
        self.logger.log_metric("beta", self.beta)
