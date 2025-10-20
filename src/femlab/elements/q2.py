import numpy as np

def shape_q2_edge(s, edge_id):
    if edge_id == 0:   xi, eta = s, -1.0
    elif edge_id == 1: xi, eta = 1.0, s
    elif edge_id == 2: xi, eta = s,  1.0
    elif edge_id == 3: xi, eta = -1.0, s
    else: raise ValueError
    N, dN_dxi, dN_deta = shape_q2(xi, eta)
    if edge_id in (0,2): dxi_ds, deta_ds = 1.0, 0.0
    else: dxi_ds, deta_ds = 0.0, 1.0
    return N, dN_dxi, dN_deta, dxi_ds, deta_ds, xi, eta


def shape_q2(xi: float, eta: float):
    # ... (N, dN_dxi, dN_deta) igual que en tu script
    def L(i, s):
        if i == -1:
            return 0.5*s*(s-1.0)
        elif i == 0:
            return (1.0 - s**2)
        elif i == 1:
            return 0.5*s*(s+1.0)
        else:
            raise ValueError
    def dL(i, s):
        if i == -1:
            return s - 0.5
        elif i == 0:
            return -2.0*s
        elif i == 1:
            return s + 0.5
        else:
            raise ValueError
    Lx = { -1: L(-1, xi), 0: L(0, xi), 1: L(1, xi) }
    Ly = { -1: L(-1, eta), 0: L(0, eta), 1: L(1, eta) }
    dLx = { -1: dL(-1, xi), 0: dL(0, xi), 1: dL(1, xi) }
    dLy = { -1: dL(-1, eta), 0: dL(0, eta), 1: dL(1, eta) }
    N = np.zeros(9); dN_dxi = np.zeros(9); dN_deta = np.zeros(9)
    ij_map = {0:(-1,-1),1:(0,-1),2:(1,-1),3:(1,0),4:(1,1),5:(0,1),6:(-1,1),7:(-1,0),8:(0,0)}
    for a in range(9):
        i,j = ij_map[a]
        N[a]      = Lx[i]*Ly[j]
        dN_dxi[a] = dLx[i]*Ly[j]
        dN_deta[a]= Lx[i]*dLy[j]
        
    return N, dN_dxi, dN_deta

def Ke_q2(xe: np.ndarray, ye: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
    # ... arma B en 2x2 Gauss y devuelve Ke (18x18)
    Ke = np.zeros((18, 18))
    g = 1.0/np.sqrt(3.0)
    gps = [(-g,-g,1.0),(g,-g,1.0),(g,g,1.0),(-g,g,1.0)]
    for (xi,eta,w) in gps:
        N, dN_dxi, dN_deta = shape_q2(xi, eta)
        J = np.zeros((2,2))
        J[0,0] = np.dot(dN_dxi,  xe); J[0,1] = np.dot(dN_deta, xe)
        J[1,0] = np.dot(dN_dxi,  ye); J[1,1] = np.dot(dN_deta, ye)
        detJ = np.linalg.det(J)
        if detJ <= 0: raise ValueError("Negative/zero detJ")
        invJ = np.linalg.inv(J)
        dN = np.vstack((dN_dxi, dN_deta))
        dN_dxdy = invJ @ dN
        B = np.zeros((3,18))
        for a in range(9):
            dNdx = dN_dxdy[0,a]; dNdy = dN_dxdy[1,a]; ia = 2*a
            B[0,ia] = dNdx; B[1,ia+1]=dNdy; B[2,ia]=dNdy; B[2,ia+1]=dNdx
        Ke += (B.T @ D @ B) * detJ * w * t
    return Ke
