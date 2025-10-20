import numpy as np
from femlab import D_plane_stress, build_q2_mesh_rect, assemble_global, apply_dirichlet, apply_neumann_edges
from femlab.postproc.von_mises import compute_cell_von_mises
from femlab.io.vtu import write_vtu,reorder_conn_to_vtk_biquadratic

def main():
    E, nu, t = 210e9, 0.3, 0.01
    xy, conn = build_q2_mesh_rect(0,1,0,1, nx=3, ny=2)
    D = D_plane_stress(E, nu)
    K, f = assemble_global(xy, conn, D, t)

    # CCs
    dirichlet = [{"node": n, "dof": "u", "value": 0.0} for n in [0,7,14]] + \
                [{"node": n, "dof": "v", "value": 0.0} for n in [0,14]]
    K, f = apply_dirichlet(K, f, dirichlet)
    
    # Neumann (ejemplo)
    neumann = [{"elem": e, "edge": 1, "tx": 1e6*t, "ty": 0.0} for e in range(len(conn)) if e % 3 == 2]
    f = apply_neumann_edges(f, conn, xy, neumann, t)

    a = np.linalg.solve(K, f)

    # Export VTU
    points = np.column_stack([xy, np.zeros(len(xy))])
    disp = np.zeros((len(xy),3)); disp[:,0]=a[0::2]; disp[:,1]=a[1::2]
    # ... arma connectivity y types (28), von_mises por celda

    vm = compute_cell_von_mises(conn, xy, a, D)

    vtk_cells = reorder_conn_to_vtk_biquadratic(conn)
    cell_types = np.full(len(vtk_cells), 28, dtype=np.uint8)
    
    root_path = "/mnt/nvme/DICYT25"
    out_file = root_path + "/case.vtu"
    write_vtu(out_file, points, vtk_cells, cell_types,point_data={"displacement": disp},cell_data={"von_mises": vm})
    print("vtu guardado en: ", out_file)
    
if __name__ == "__main__":
    main()

