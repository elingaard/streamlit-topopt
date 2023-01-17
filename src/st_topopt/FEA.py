from typing import Tuple

import numpy as np
import scipy.sparse as sp


class QuadMesh:
    """Class for storing the attributes of a rectangular finite element mesh

    Attributes:
    self.nelx(int): number of elements in x-direction
    self.nely(int): number of elements in y-direction
    self.nele(int): number of elements in mesh
    self.nnodes(int): number of nodes in mesh
    self.ndof(int): number of degrees-of-freedom in mesh
    self.XY(nnodes x 2 int matrix): mesh node coordinates
    self.IX(nele x 4 int matrix): element connectivity matrix
    self.edof_mat(nele x 8 int matrix): matrix containing degrees-of-freedom for each
    element

    Mesh conventions:
    - Node numbering - column-wise, starting at the top
    0---3---6
    |   |   |
    1---4---7
    |   |   |
    2---5---8
    - Element coordinate system
    (-1,1)----(1,1)
    |             |
    |    (0,0)    |
    |             |
    (-1,-1)--(1,-1)
    - Face counts
    *---0---*
    |       |
    3       1
    |       |
    *---2---*
    """

    def __init__(self, nelx: int, nely: int):
        self.nelx = nelx
        self.nely = nely
        self.nele = nelx * nely
        self.nnodes = (nelx + 1) * (nely + 1)
        self.ndof = 2 * self.nnodes
        Y_grid, X_grid = np.meshgrid(range(nely + 1), range(nelx + 1))
        self.XY = np.vstack((X_grid.flatten(), Y_grid.flatten())).T
        self.IX = self.get_connectivity_matrix()
        self.edof_mat = self.get_edof_matrix()

    def get_connectivity_matrix(self) -> np.ndarray:
        """Function for creating the connectivity matrix between elements and
        nodes in the mesh

        Returns:
        IX(nele x 4 int matrix): element connectivity matrix

        """
        elx, ely = np.meshgrid(np.arange(self.nelx), np.arange(self.nely))
        elx_vec = elx.flatten(order="F")
        ely_vec = ely.flatten(order="F")
        node0 = (self.nely + 1) * elx_vec + (ely_vec + 1)
        node1 = (self.nely + 1) * (elx_vec + 1) + (ely_vec + 1)
        node2 = (self.nely + 1) * (elx_vec + 1) + (ely_vec)
        node3 = (self.nely + 1) * elx_vec + ely_vec
        IX = np.vstack((node0, node1, node2, node3)).T

        return IX

    def get_edof_matrix(self) -> np.ndarray:
        """Function for creating the element degrees-of-freedom matrix which
        holds the degrees-of-freedom for each element in the mesh

        Returns:
        edof_mat(nele x 8 int matrix): matrix containing degrees-of-freedom for each
        element

        """
        edof_mat = np.zeros((self.nele, 8), dtype=int)
        edof_mat[:, 0:8:2] = self.IX * 2
        edof_mat[:, 1:8:2] = self.IX * 2 + 1

        return edof_mat

    def grad_shape_func(self, xi: float, eta: float) -> np.ndarray:
        """Function for evaluation the differentiated rectangular shape function at a
        given point xi, eta

        Args:
        xi(float): element evaluation point in x
        eta(float): element evaluation point in y

        Returns:
        dN_mat (4x8 float matrix): differentiated shape function matrix

        """
        # Differentiated shape function values
        dN_xi = 1 / 4 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_eta = 1 / 4 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])

        # Differentiated shape function matrix
        dN_mat = np.zeros((4, 8))
        dN_mat[0, 0:8:2] = dN_xi
        dN_mat[1, 0:8:2] = dN_eta
        dN_mat[2, 1:8:2] = dN_xi
        dN_mat[3, 1:8:2] = dN_eta

        return dN_mat

    def dof2nodeid(self, dof: int) -> int:
        """Function for getting the node associated with a given degree-of-freedom

        Args:
        dof(int): degree-of-freedom

        Returns:
        node(int): node index

        """
        return np.floor(dof / 2).astype(int)


class LinearElasticity:
    """Class for storing functions and attributes related to solving linear elasticity
    problems

    Attributes:
    self.mesh(object): finite element mesh object
    self.poisson_ratio(float): poisson ratio of the desired material
    self.youngs_modulus(float): youngs modulus of the desired material
    self.Emin(float): minimum value of youngs modulus, used for computational stability
    self.Emax(float): maximum value of youngs modulus, used for computational stability
    self.penal(float): stiffness penalty parameter for intermediate material densities
    self.fixed_dofs(Nx1 int array): fixed degrees-of-freedom
    self.load(ndofx1 int array): load vector
    self.stiffness_matrix(ndof x ndof float matrix): stiffness matrix (only available
    after building stiffness matrix)
    self.displacement(ndof x 1 float array): displacement vector (only available
    after solving system)

    """

    def __init__(
        self,
        mesh: QuadMesh,
        poisson_ratio: float = 0.3,
        youngs_modulus: float = 1.0,
    ):
        self.mesh = mesh
        self.poisson_ratio = poisson_ratio
        self.youngs_modulus = youngs_modulus
        self.Emin = 1e-9 * self.youngs_modulus
        self.Emax = self.youngs_modulus
        self.fixed_dofs = []
        self.load = np.zeros(mesh.ndof)
        self.stiffness_matrix = None
        self.displacement = None
        self.Cmat = self.constitutive_stress_matrix()
        self.B0 = self.strain_displacement_matrix()
        self.KE = self.element_stiffness_matrix()

    @staticmethod
    def is_integer_np_array(arr):
        assert isinstance(arr, np.ndarray)
        assert issubclass(arr.dtype.type, np.integer)

    def element_stiffness_matrix(self) -> np.ndarray:
        """Function for creating the analytical element stiffness matrix

        Returns:
        KE(8x8 float matrix): element stiffness matrix

        """
        ke = np.array(
            [
                1 / 2 - self.poisson_ratio / 6,
                1 / 8 + self.poisson_ratio / 8,
                -1 / 4 - self.poisson_ratio / 12,
                -1 / 8 + 3 * self.poisson_ratio / 8,
                -1 / 4 + self.poisson_ratio / 12,
                -1 / 8 - self.poisson_ratio / 8,
                self.poisson_ratio / 6,
                1 / 8 - 3 * self.poisson_ratio / 8,
            ]
        )

        ele_stiff_mat = np.array(
            [
                [ke[0], ke[1], ke[2], ke[3], ke[4], ke[5], ke[6], ke[7]],
                [ke[1], ke[0], ke[7], ke[6], ke[5], ke[4], ke[3], ke[2]],
                [ke[2], ke[7], ke[0], ke[5], ke[6], ke[3], ke[4], ke[1]],
                [ke[3], ke[6], ke[5], ke[0], ke[7], ke[2], ke[1], ke[4]],
                [ke[4], ke[5], ke[6], ke[7], ke[0], ke[1], ke[2], ke[3]],
                [ke[5], ke[4], ke[3], ke[2], ke[1], ke[0], ke[7], ke[6]],
                [ke[6], ke[3], ke[4], ke[1], ke[2], ke[7], ke[0], ke[5]],
                [ke[7], ke[2], ke[1], ke[4], ke[3], ke[6], ke[5], ke[0]],
            ]
        )
        ele_stiff_mat *= self.youngs_modulus / (1 - self.poisson_ratio**2)

        return ele_stiff_mat

    def strain_displacement_matrix(self) -> np.ndarray:
        """Function for creating strain-displacement matrix based on shape functions

        Returns:
        B0(3x8 float matrix): strain-displacement matrix
        """
        # matrix used for mapping shape function derivatives into strains
        L = np.zeros((3, 4))
        L[0, 0] = 1
        L[1, 3] = 1
        L[2, 1:3] = 1
        # evaluation point
        xi = 0
        eta = 0
        # differentiated shape functions evaluated in (xi,eta)
        dN_mat = self.mesh.grad_shape_func(xi, eta)
        # mapping back to global coordinates
        jac = np.array([[0.5, 0], [0, 0.5]])
        detJ = np.linalg.det(jac)
        invJ = 1 / detJ * np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]])

        # mapping matrix between rectangular element space coordinates and global
        # coordinates
        G = np.zeros((4, 4))
        G[:2, :2] = invJ
        G[-2:, -2:] = invJ
        # strain displacement matrix
        strain_disp_mat = np.matmul(np.matmul(L, G), dN_mat)

        return strain_disp_mat

    def constitutive_stress_matrix(self) -> np.ndarray:
        """Function for creating the constitutive stiffness matrix

        Returns:
        Cmat(3x3 float matrix): constitutive stiffness matrix
        """
        Cmat = np.array(
            [
                [1, self.poisson_ratio, 0],
                [self.poisson_ratio, 1, 0],
                [0, 0, (1 - self.poisson_ratio) / 2],
            ]
        )
        Cmat *= self.youngs_modulus / (1 - self.poisson_ratio**2)
        return Cmat

    def stiffness_matrix_assembly(
        self, rho: np.ndarray, penal: float, sparse: bool = True
    ) -> np.ndarray:
        """Assembles the global stiffness from the local element matrices

        Args:
        rho(nely x nelx float matrix): density matrix
        penal(float): penalty of intermediate densities
        sparse(bool): matrix type

        Returns:
        stiffness_matrix(ndof x ndof float matrix): global stiffness matrix

        """
        rho = rho.flatten(order="F")
        if sparse is False:
            self.stiffness_matrix = np.zeros((self.mesh.ndof, self.mesh.ndof))
            mat_idxs = np.indices((8, 8))
            iK = mat_idxs[0].flatten()
            jK = mat_idxs[1].flatten()
            # assemble stiffness matrix
            for ele in range(len(self.mesh.edof_mat)):
                edof = self.mesh.edof_mat[ele]
                self.stiffness_matrix[edof[iK], edof[jK]] += (
                    self.Emin + (rho[ele]) ** penal * (self.Emax - self.Emin)
                ) * self.KE[iK, jK]
        elif sparse is True:
            iK = np.kron(self.mesh.edof_mat, np.ones((8, 1))).flatten()
            jK = np.kron(self.mesh.edof_mat, np.ones((1, 8))).flatten()
            sK = (
                (self.KE.flatten()[np.newaxis]).T
                * (self.Emin + (rho) ** penal * (self.Emax - self.Emin))
            ).flatten(order="F")
            self.stiffness_matrix = sp.coo_matrix(
                (sK, (iK, jK)), shape=(self.mesh.ndof, self.mesh.ndof)
            ).tocsc()

        return self.stiffness_matrix

    def solve_(
        self, rho: np.ndarray, penal, unit_load: bool = False, sparse: bool = True
    ) -> np.ndarray:
        """Solves the linear system Ku=F

        Args:
        rho(nely x nelx float matrix): density matrix
        penal(float): penalty of intermediate densities
        unit_load(bool): normalize load to unit magnitude
        sparse(bool): matrix type

        Returns:
        U(ndof x 1 float array): displacement vector

        """

        # check whether load and boundary conditions have been applied
        n_loaded_nodes = len(np.nonzero(self.load)[0])
        if (len(self.fixed_dofs) == 0) or (n_loaded_nodes == 0):
            raise AttributeError(
                "Please insert load and boundary conditions before solving the system"
            )

        # normalize load if flag is true
        if unit_load is True:
            load_sum = np.sum(np.abs(self.load))
            self.load /= load_sum

        # assemble the stiffness matrix
        if sparse is False:
            self.stiffness_matrix_assembly(rho, penal, sparse=False)
            # apply BC using zero-one method
            self.stiffness_matrix[self.fixed_dofs, :] = 0
            self.stiffness_matrix[:, self.fixed_dofs] = 0
            self.stiffness_matrix[self.fixed_dofs, self.fixed_dofs] = 1
            self.load[self.fixed_dofs] = 0
            # solve system
            disp = np.linalg.solve(self.stiffness_matrix, self.load)
        elif sparse is True:
            self.stiffness_matrix_assembly(rho, penal, sparse=True)
            # get free dofs
            all_dofs = np.arange(2 * (self.mesh.nelx + 1) * (self.mesh.nely + 1))
            free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)
            K_free = self.stiffness_matrix[free_dofs, :][:, free_dofs]
            # only solve system for free dofs
            disp = np.zeros(self.mesh.ndof)
            disp[free_dofs] = sp.linalg.spsolve(K_free, self.load[free_dofs])

        # store displacement vector for later use (strain energy, stresses etc.)
        self.displacement = disp.copy()

        return disp

    def insert_node_boundaries(self, node_ids: np.ndarray, axis: int) -> None:
        """Modifies the fixed_dofs vector to incorporate new point boundary condition(s)

        Args:
        node_ids(np.array): integer array with node ids
        axis(int): direction along which the boundary condition is enforced

        """
        self.is_integer_np_array(node_ids)
        assert axis < 2

        bound_dofs = node_ids * 2 + (1 * axis)
        self.fixed_dofs = np.union1d(self.fixed_dofs, bound_dofs)

    def insert_node_load(self, node_id: int, load_vec: tuple) -> None:
        """Modifies the load vector to incorporate a new point load

        Args:
        node_id(int): node in which the load is applied
        load_vec(tuple): load magnitude in x and y direction

        """
        assert type(node_id) == int
        assert len(load_vec) == 2

        self.load[node_id * 2] += load_vec[0]
        self.load[node_id * 2 + 1] += load_vec[1]

    def pre_integrated_face_force_vectors(self) -> np.ndarray:
        """Function for creating the pre-integrated force vectors for loading of
        element faces

        Local node numbering (different from global)
        1-------2
        |       |
        |       |
        4-------3

        Returns:
        face_force_vecs(8x8 float array): element face forces

        """
        # unit dimensions for element
        a = 0.5
        b = 0.5
        # initialize traction to unity
        tr11 = 1
        tr22 = 1

        f0x_face1 = np.zeros(8)
        f0x_face1[[4, 6]] = tr11 * a

        f0y_face1 = np.zeros(8)
        f0y_face1[[5, 7]] = tr22 * a

        f0x_face2 = np.zeros(8)
        f0x_face2[[2, 4]] = tr11 * b

        f0y_face2 = np.zeros(8)
        f0y_face2[[3, 5]] = tr22 * b

        f0x_face3 = np.zeros(8)
        f0x_face3[[0, 2]] = tr11 * a

        f0y_face3 = np.zeros(8)
        f0y_face3[[1, 3]] = tr22 * a

        f0x_face4 = np.zeros(8)
        f0x_face4[[0, 6]] = tr11 * b

        f0y_face4 = np.zeros(8)
        f0y_face4[[1, 7]] = tr22 * b

        face_force_vecs = np.vstack(
            (
                f0x_face1,
                f0y_face1,
                f0x_face2,
                f0y_face2,
                f0x_face3,
                f0y_face3,
                f0x_face4,
                f0y_face4,
            )
        )
        return face_force_vecs

    def pre_integrated_element_force_vectors(self) -> np.ndarray:
        """Function for creating the pre-integrated force vectors for loading of
        entire elements

        Returns:
        ele_force_vecs(2x8 float array): element forces

        """
        # unit dimensions for element
        a = 0.5
        b = 0.5

        # initialize body force to unity
        bf11 = 1
        bf22 = 1

        f0x_element = np.zeros(8)
        f0x_element[[0, 2, 4, 6]] = a * b * bf11

        f0y_element = np.zeros(8)
        f0y_element[[1, 3, 5, 7]] = a * b * bf22

        ele_force_vecs = np.vstack((f0x_element, f0y_element))

        return ele_force_vecs

    def insert_face_forces(
        self, ele_ids: np.ndarray, ele_face: int, load_vec: tuple
    ) -> None:
        """
        Given a set of element ids add a force specified by 'load_vec' to the specifed
        'ele_face' for each element

        Args:
        ele_ids(integer array): array of element ids for which a face force is applied
        ele_face(int): integer indicating which face the load is applied to
        load_vec(tuple): load magnitude in x and y direction

        """
        self.is_integer_np_array(ele_ids)
        assert ele_face < 4
        assert len(load_vec) == 2

        face_force_vecs = self.pre_integrated_face_force_vectors()
        f0x_face, f0y_face = face_force_vecs[ele_face * 2 : (ele_face + 1) * 2]
        fvec = f0x_face * load_vec[0] + f0y_face * load_vec[1]
        load_dofs = self.mesh.edof_mat[ele_ids].flatten()
        self.load[load_dofs] += np.tile(fvec, len(ele_ids))

    def insert_element_forces(self, ele_ids: np.ndarray, load_vec: tuple) -> None:
        """
        Given a set of element ids add a force specified by 'load_vec' to the specifed
        element for each element

        Args:
        ele_ids(int array): array of element ids for which an element force is applied
        load_vec(tuple): load magnitude in x and y direction

        """
        self.is_integer_np_array(ele_ids)
        assert len(load_vec) == 2

        f0x_element, f0y_element = self.pre_integrated_element_force_vectors()
        fvec = f0x_element * load_vec[0] + f0y_element * load_vec[1]
        load_dofs = self.mesh.edof_mat[ele_ids].flatten()
        self.load[load_dofs] += np.tile(fvec, len(ele_ids))

    def compute_compliance(self, disp: np.ndarray) -> float:
        """Calculates the compliance (U^T*K*U) using the displacement vector from the
        solved system

        Args:
        disp(float array): displacement vector

        Returns:
        compliance(float): compliance value

        """

        if sp.issparse(self.stiffness_matrix) == True:
            compliance = self.stiffness_matrix.dot(disp).dot(disp)
        else:
            compliance = (disp @ self.stiffness_matrix) @ disp
        return compliance.item()

    def compute_element_strains(self, disp: np.ndarray) -> np.ndarray:
        """Compute the strain-tensor in each element

        Args:
        disp(float array): displacement vector

        Returns:
        epsilon_mat(3 x nely x nelx) float matrix): epsilon_11, epsilon_22, epsilon_12
        """

        epsilon_mat = self.B0 @ disp[self.mesh.edof_mat].T
        # reshape element-wise strains to matrix
        epsilon_mat = epsilon_mat.reshape(
            (3, self.mesh.nely, self.mesh.nelx), order="F"
        )
        # divide by 2 since epsilon vector is defined as [eps_11,eps_22,2*eps_12]
        epsilon_mat[2] /= 2

        return epsilon_mat

    def compute_element_stresses(self, disp: np.ndarray) -> np.ndarray:
        """Compute the stress-tensor in each element

        Args:
        disp(float array): displacement vector

        Returns:
        sigma_mat(3 x nely x nelx) float matrix): sigma_11, sigma_22, sigma_12
        """

        # reshape element-wise strains to matrix
        sigma_mat = self.Cmat @ (self.B0 @ disp[self.mesh.edof_mat].T)
        sigma_mat = sigma_mat.reshape((3, self.mesh.nely, self.mesh.nelx), order="F")

        return sigma_mat

    def compute_strain_energy(self, disp: np.ndarray) -> np.ndarray:
        """Compute the strain-energy in each element

        Args:
        disp(float array): displacement vector

        Returns:
        strain_density_mat(nely x nelx float matrix): strain-density in each element
        """

        epsilon_mat = self.compute_element_strains(disp)
        sigma_mat = self.compute_element_stresses(disp)
        strain_density_mat = (
            sigma_mat[0] * epsilon_mat[0]
            + sigma_mat[1] * epsilon_mat[1]
            + (2 * sigma_mat[2] * epsilon_mat[2])
        )
        strain_density_mat *= 1 / 2

        return strain_density_mat

    def compute_von_mises_stresses(self, disp: np.ndarray) -> np.ndarray:
        """Compute the von mises stress in each element

        Args:
        disp(float array): displacement vector

        Returns:
        sigma_vm(nely x nelx float matrix): von mises stress in each element
        """

        sigma_mat = self.compute_element_stresses(disp)
        sigma_vm = (
            sigma_mat[0] ** 2
            + sigma_mat[1] ** 2
            - (sigma_mat[0] * sigma_mat[1]) + (3 * sigma_mat[2] ** 2)
        )
        sigma_vm = np.sqrt(sigma_vm)
        return sigma_vm

    def compute_principal_stresses(
        self, disp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the principal stresses and direction in each element

        Args:
        disp(float array): displacement vector

        Returns:
        sigma_1(nely x nelx float matrix): first principal stress
        sigma_2(nely x nelx float matrix): second principal stress
        psi_1(nely x nelx float matrix): first principal stress direction
        psi_2(nely x nelx float matrix): second principal stress direction
        """

        sigma_mat = self.compute_element_stresses(disp)
        first_term = 1 / 2 * (sigma_mat[0] + sigma_mat[1])
        second_term = np.sqrt(
            ((sigma_mat[0] - sigma_mat[1]) / 2) ** 2 + sigma_mat[2] ** 2
        )
        sigma1 = first_term + second_term
        sigma2 = first_term - second_term
        psi1 = np.arctan2(-2 * sigma_mat[2], sigma_mat[0] - sigma_mat[1]) / 2
        psi2 = psi1 - np.pi / 2
        return sigma1, sigma2, psi1, psi2
