import numpy as np

def build_q2_mesh_rect(x0, x1, y0, y1, nx, ny):
    # ... devuelve xy (N×2) y conn (nelems×9)
    nxn = 2*nx + 1; nyn = 2*ny + 1
    xs = np.linspace(x0, x1, nxn); ys = np.linspace(y0, y1, nyn)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    def node_id(ix, iy): return iy*nxn + ix
    conn = []
    for ey in range(ny):
        for ex in range(nx):
            ix0 = 2*ex; iy0=2*ey
            n0 = node_id(ix0,     iy0    )
            n1 = node_id(ix0 + 1, iy0    )
            n2 = node_id(ix0 + 2, iy0    )
            n3 = node_id(ix0 + 2, iy0 + 1)
            n4 = node_id(ix0 + 2, iy0 + 2)
            n5 = node_id(ix0 + 1, iy0 + 2)
            n6 = node_id(ix0,     iy0 + 2)
            n7 = node_id(ix0,     iy0 + 1)
            n8 = node_id(ix0 + 1, iy0 + 1)
            conn.append([n0,n1,n2,n3,n4,n5,n6,n7,n8])
    #return np.array(xy), np.array(conn, dtype=int), nx, ny
    return xy, conn
