import numpy as np
import xarray as xr
from juliacall import Main as jl

jl.seval("using QSpaceTools")
QST = jl.QSpaceTools

NCH1, NCH2, NFRAMES = 32, 40, 5
GRID = (24, 26, 28)

geom = QST.Geometry(
    sample_axes=("y-", "x+", "z+"), detector_axes=("y-",), image_axes=("y-", "z-"),
    beam_direction=(1.0, 0.0, 0.0), sample_normal=(0.0, 0.0, 1.0), sample_faceup="z+",
    pixel_size=(0.2, 0.2), center=(NCH1 // 2, NCH2 // 2), shape=(NCH1, NCH2),
    distance=500.0, wavelength=QST.energy2wavelength(8000.0))
kwargs = dict(sample_angles=(17.0, 0.05, -0.10), detector_angles=(35.5,))

# Create row-major test data
frames = np.random.rand(NFRAMES, NCH2, NCH1)
reference = np.asarray(jl.parent(QST.rsm(jl.convert(jl.Array[jl.Float64, 3], frames.T), geom,
                                         gridder_size=GRID, output="volume",
                                         **kwargs)))
inputs = [frames, xr.DataArray(frames, dims=("frame", "ss", "fs"))]


def test_rsm():
    for py_frames in inputs:
        got = QST.rsm(py_frames, geom, gridder_size=GRID, output="volume", **kwargs)
        np.testing.assert_array_equal(np.asarray(jl.parent(got)), reference)

def test_rsm_bang():
    for py_frames in inputs:
        out = QST.allocate_output(geom, GRID, output="volume")
        QST.rsm_b(out, py_frames, geom, **kwargs)
        np.testing.assert_array_equal(np.asarray(out), reference)
