import numpy as np
import xml.sax.saxutils as sx

def write_vtu(filename: str, points: np.ndarray, cells: list[np.ndarray],
              cell_types: np.ndarray, point_data=None, cell_data=None):
    # ... escritor VTU “unstructured grid”
    N = points.shape[0]; nc = len(cells)
    conn_all = np.concatenate(cells)
    offsets = np.cumsum([len(c) for c in cells])
    import xml.sax.saxutils as sx
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{N}" NumberOfCells="{nc}">\n')
        if point_data:
            f.write('      <PointData>\n')
            for name, arr in point_data.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    f.write(f'        <DataArray type="Float64" Name="{sx.escape(name)}" format="ascii" NumberOfComponents="1">\n')
                    f.write("          " + " ".join(map(str, arr.tolist())) + "\n")
                    f.write('        </DataArray>\n')
                elif arr.ndim == 2 and arr.shape[1] in (3,):
                    f.write(f'        <DataArray type="Float64" Name="{sx.escape(name)}" format="ascii" NumberOfComponents="{arr.shape[1]}">\n')
                    flat = arr.reshape(-1)
                    f.write("          " + " ".join(map(str, flat.tolist())) + "\n")
                    f.write('        </DataArray>\n')
            f.write('      </PointData>\n')
        else:
            f.write('      <PointData/>\n')
        if cell_data:
            f.write('      <CellData>\n')
            for name, arr in cell_data.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    f.write(f'        <DataArray type="Float64" Name="{sx.escape(name)}" format="ascii" NumberOfComponents="1">\n')
                    f.write("          " + " ".join(map(str, arr.tolist())) + "\n")
                    f.write('        </DataArray>\n')
                elif arr.ndim == 2 and arr.shape[1] in (3,):
                    f.write(f'        <DataArray type="Float64" Name="{sx.escape(name)}" format="ascii" NumberOfComponents="{arr.shape[1]}">\n')
                    flat = arr.reshape(-1)
                    f.write("          " + " ".join(map(str, flat.tolist())) + "\n")
                    f.write('        </DataArray>\n')
            f.write('      </CellData>\n')
        else:
            f.write('      <CellData/>\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + " ".join(map(str, points.reshape(-1).tolist())) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(map(str, conn_all.tolist())) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(map(str, offsets.tolist())) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(map(str, cell_types.tolist())) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

def reorder_conn_to_vtk_biquadratic(conn):
    map_idx = [0,2,4,6,1,3,5,7,8]
    vtk_conn = [np.array([elem[i] for i in map_idx], dtype=int) for elem in conn]
    return vtk_conn
