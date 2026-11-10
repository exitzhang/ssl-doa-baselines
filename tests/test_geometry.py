import numpy as np

from ssl_doa_baselines.geometry import angles_to_vectors, vectors_to_angles


def test_zenith_convention():
    # Horizontal source: elevation 90 in the zenith convention.
    v = angles_to_vectors(np.array([[30.0, 90.0], [120.0, 90.0]]))
    assert np.allclose(v[0], [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0], atol=1e-6)
    assert abs(v[0, 2]) < 1e-6


def test_roundtrip_azimuth():
    angles = np.array([[0.0, 90.0], [45.0, 90.0], [90.0, 90.0],
                       [135.0, 90.0], [180.0, 90.0]])
    back = vectors_to_angles(angles_to_vectors(angles))
    assert np.allclose(back[:, 0], angles[:, 0], atol=1e-4)
    assert np.allclose(back[:, 1], 90.0, atol=1e-4)


def test_empty():
    assert angles_to_vectors(np.zeros((0, 2))).shape == (0, 3)
