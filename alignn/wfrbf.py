import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list
from numba import njit, vectorize
from numpy import conjugate, cos, exp, pi, sqrt
from scipy.special import eval_genlaguerre, factorial, lpmv
from torch_scatter import scatter


@vectorize("int64(float64)")
def factorial(n):
    LOOKUP_TABLE = np.array(
        [
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            39916800,
            479001600,
            6227020800,
            87178291200,
            1307674368000,
            20922789888000,
            355687428096000,
            6402373705728000,
            121645100408832000,
            2432902008176640000,
        ],
        dtype="int64",
    )

    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[int(n)]


@vectorize("complex128(int64,float64,float64,float64,float64,float64,int64)")
def psi(n, l, m, r, phi, theta, Z):
    x0 = Z * r / n
    x1 = 2 * x0
    x2 = -l + n - 1
    x3 = 2 * l
    x4 = factorial(l - m) / factorial(l + m)
    return (
        x1 ** l
        * sqrt(Z ** 3 * factorial(x2) / (n ** 4 * factorial(l + n)))
        * sqrt(x3 * x4 + x4)
        * exp(-x0)
        * exp(1j * m * phi)
        * eval_genlaguerre(x2, x3 + 1, x1)
        * lpmv(m, l, cos(theta))
        / sqrt(pi)
    )


@njit
def add_self_loops(edge_index, num_nodes, edge_attr=None, fill_value=1.0):
    loop_index = np.arange(0, num_nodes, dtype=np.int64)
    loop_index = np.vstack((loop_index, loop_index))

    if edge_attr is not None:
        loop_attr = np.full((num_nodes,) + edge_attr.shape[1:], fill_value)

        edge_attr = np.vstack((edge_attr, loop_attr))

    edge_index = np.hstack((edge_index, loop_index))
    return edge_index, edge_attr


@njit
def compute_spherical(edge_index, pos, num_nodes):
    (row, col), pos = edge_index, pos

    cart = pos[col] - pos[row]

    rho = np.array([np.linalg.norm(a, 2) for a in cart])

    theta = np.arctan2(cart[..., 1], cart[..., 0]).reshape(-1, 1)
    theta = theta + (theta < 0) * (2 * pi)

    phi = np.arccos(cart[..., 2] / rho.reshape(-1)).reshape(-1, 1)

    theta = theta / (2 * pi)
    phi = phi / pi

    spher = np.hstack((theta, phi))

    edge_index, edge_spherical = add_self_loops(
        edge_index, edge_attr=spher, num_nodes=num_nodes, fill_value=0.0
    )

    return edge_index, edge_spherical


@njit(
    "complex128[:](int64, float64, int64, float64, float64, float64, int64, int64)",
    nogil=True,
)
def compute_wf(n, l, multiplie, atom_r, phi, theta, z, num_features):
    spaces = np.linspace(0, atom_r / 100, num_features)
    wf = psi(n, l, np.expand_dims(np.arange(-l, l + 1), 1), spaces, phi, theta, z).sum(
        0
    )
    wf = conjugate(wf) * wf

    return wf * spaces ** 2 * multiplie


class WFRBF:
    NOBLE = {
        "He": "1s2",
        "Ne": "1s2 2s2 2p6",
        "Ar": "1s2 2s2 2p6 3s2 3p6",
        "Kr": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6",
        "Xe": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6",
        "Rn": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 6s2 4f14 5d10 6p6",
    }
    ORBITALS = ("s", "p", "d", "f", "g", "h", "i", "j", "k")

    def __init__(self, molecules_df, num_features=128):
        self.molecules_df = molecules_df
        self.num_features = num_features
        self.electronic_configuration = self._get_electronic_configuration()
        self.atomic_radius = np.array(molecules_df["atomic_radius"].values)

    def __call__(self, z, pos, **atoms_kwargs):
        zs, pos = np.array(z, dtype=np.int64), np.array(pos)
        atoms = self._data2atoms(zs, pos, **atoms_kwargs)
        n_index, c_index = neighbor_list("ij", atoms, natural_cutoffs(atoms))
        mult = 1.01
        while len(n_index) == 0:
            # print(mult)
            n_index, c_index = neighbor_list(
                "ij", atoms, natural_cutoffs(atoms, mult=mult)  # type: ignore
            )
            mult += 0.01
        edge_index = np.vstack((n_index, c_index))

        electronic_configurations = self.electronic_configuration[zs - 1]
        atomic_radiuses = self.atomic_radius[zs - 1]
        wf_edge_index, edge_spherical = compute_spherical(edge_index, pos, len(z))

        (row, _) = wf_edge_index
        (thetas, phis) = edge_spherical.T
        zs = zs[row]
        electronic_configurations = electronic_configurations[row]
        atomic_radiuses = atomic_radiuses[row]

        wf_rbf = np.zeros((zs.shape[0], self.num_features), dtype=np.float32)
        # print(wf_rbf.size())
        for i, (z, e_c, a_r, phi, theta) in enumerate(
            zip(zs, electronic_configurations, atomic_radiuses, phis, thetas)
        ):
            wf_rbf[i] = self.get_wf_rbf(z, e_c, a_r, phi, theta)

        wf_scatter = scatter(
            torch.tensor(wf_rbf), index=torch.tensor(row), dim=0, reduce="mean"
        )

        return wf_scatter, wf_rbf, wf_edge_index

    def get_wf_rbf(
        self, z: int, e_congiguration: str, atom_r: float, phi: float, theta: float
    ):
        e_congiguration = e_congiguration.strip()
        if e_congiguration.startswith("["):
            e_congiguration = "".join(
                self.NOBLE[e_congiguration[1:3]] + e_congiguration[4:]
            )
        e_cs = e_congiguration.split(" ")
        wf_rbf = np.zeros(self.num_features, dtype=np.float32)
        for e_c in e_cs:
            n = int(e_c[0])
            l = float(self.ORBITALS.index(e_c[1]))
            multiplie = int(e_c[2]) if len(e_c) == 3 else 1

            wf_rbf += compute_wf(
                n, l, multiplie, atom_r, phi, theta, z, self.num_features
            ).astype(np.float32)
        return wf_rbf

    def _get_electronic_configuration(self):
        e_cs = self.molecules_df["electronic_configuration"].values
        e_cs_lst = []
        for e_c in e_cs:
            if e_c.startswith("["):
                e_c = "".join(self.NOBLE.get(e_c[1:3]) + e_c[4:])
            e_cs_lst.append(e_c)
        return np.array(e_cs_lst)

    @staticmethod
    def _data2atoms(z, pos, cell=None, pbc=None, **atoms_kwargs):
        atoms = Atoms(numbers=z, positions=pos, cell=cell, pbc=pbc, **atoms_kwargs)
        return atoms


if __name__ == "__main__":
    import torch
    from mendeleev.fetch import fetch_table
    from torch_geometric.data import Data

    elements_df = fetch_table(
        "elements",
        columns=[
            "atomic_number",
            "atomic_radius",
            "electronic_configuration",
            "symbol",
        ],
    ).iloc[
        :84
    ]  # type: ignore
    wf_rbf = WFRBF(elements_df, 128)

    d = Data(z=torch.arange(1, 10, 1), pos=torch.randn((9, 3)))

    print(wf_rbf(d.z, d.pos))
