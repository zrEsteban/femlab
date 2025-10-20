import numpy as np
from .elements.q2 import Ke_q2,shape_q2_edge
from .mesh.rect_q2 import build_q2_mesh_rect


def assemble_global(xy, conn, D, t=1.0):
    ndofs = 2*len(xy); K = np.zeros((ndofs, ndofs)); f = np.zeros(ndofs)
    # ... loop elementos, scatter Ke en K
    for nodes in conn:
        xe = xy[nodes,0]; ye = xy[nodes,1]
        Ke = Ke_q2(xe, ye, D, t)
        dofs = np.zeros(18, dtype=int)
        for i_loc, n in enumerate(nodes):
            dofs[2*i_loc] = 2*n; dofs[2*i_loc+1]=2*n+1
        K[np.ix_(dofs,dofs)] += Ke
    return K, f


def apply_dirichlet(K, f, bcs):
    # ... elimina filas/cols, set RHS
    for bc in bcs:
        node=bc["node"]; dof_char=bc["dof"].lower(); val=bc["value"]
        dof = 2*node if dof_char=="u" else 2*node+1
        K[dof,:]=0.0; K[:,dof]=0.0; K[dof,dof]=1.0; f[dof]=val

    return K, f

def edge_gauss_1d():
    g = 1.0/np.sqrt(3.0); return [(-g,1.0),(g,1.0)]
    
def apply_neumann_edges(f, conn, xy, edges, t=1.0):
    # ... integra tracción en aristas
    for item in edges:
        e_id = item["elem"]; edge_id=item["edge"]; tx=item["tx"]; ty=item["ty"]
        nodes = conn[e_id]; xe=xy[nodes,0]; ye=xy[nodes,1]
        fe = np.zeros(18)
        for s, w in edge_gauss_1d():
            N, dN_dxi, dN_deta, dxi_ds, deta_ds, xi, eta = shape_q2_edge(s, edge_id)
            dx_dxi = np.dot(dN_dxi, xe); dx_deta = np.dot(dN_deta, xe)
            dy_dxi = np.dot(dN_dxi, ye); dy_deta = np.dot(dN_deta, ye)
            dx_ds = dx_dxi*dxi_ds + dx_deta*deta_ds
            dy_ds = dy_dxi*dxi_ds + dy_deta*deta_ds
            jac_line = np.sqrt(dx_ds**2 + dy_ds**2)
            for a in range(9):
                ia = 2*a
                fe[ia  ] += N[a]*tx*jac_line*w*t
                fe[ia+1] += N[a]*ty*jac_line*w*t
        dofs = np.zeros(18, dtype=int)
        for i_loc, n in enumerate(nodes):
            dofs[2*i_loc]=2*n; dofs[2*i_loc+1]=2*n+1
        f[dofs] += fe

    return f
