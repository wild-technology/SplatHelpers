"""Unit tests for critical math operations across MSF modules."""

import numpy as np
import pytest


class TestSigmoid:
    """Test sigmoid / inverse-sigmoid round-trip for opacity conversion."""

    def test_round_trip(self):
        from msf.splat_filter.ply_io import sigmoid, inv_sigmoid
        raw = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        np.testing.assert_allclose(inv_sigmoid(sigmoid(raw)), raw, atol=1e-10)

    def test_sigmoid_range(self):
        from msf.splat_filter.ply_io import sigmoid
        raw = np.linspace(-20, 20, 1000)
        s = sigmoid(raw)
        assert np.all(s >= 0.0)
        assert np.all(s <= 1.0)

    def test_sigmoid_known_values(self):
        from msf.splat_filter.ply_io import sigmoid
        np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5])

    def test_inv_sigmoid_boundary(self):
        from msf.splat_filter.ply_io import inv_sigmoid
        # Should handle values very close to 0 and 1 without crashing
        result = inv_sigmoid(np.array([1e-10, 1 - 1e-10]))
        assert np.all(np.isfinite(result))


class TestSHDCtoRGB:
    """Test spherical harmonics DC band to RGB conversion."""

    def test_known_conversion(self):
        from msf.splat_filter.ply_io import sh_dc_to_rgb
        C0 = 0.28209479177387814
        # f_dc = 0 should give rgb = 0.5
        rgb = sh_dc_to_rgb(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(rgb, [[0.5, 0.5, 0.5]], atol=1e-10)

    def test_positive_dc(self):
        from msf.splat_filter.ply_io import sh_dc_to_rgb
        C0 = 0.28209479177387814
        # f_dc = 1/C0 should give rgb = 1.5 → clipped to 1.0
        val = 1.0 / C0
        rgb = sh_dc_to_rgb(np.array([val]), np.array([val]), np.array([val]))
        np.testing.assert_allclose(rgb, [[1.0, 1.0, 1.0]], atol=1e-10)

    def test_output_shape(self):
        from msf.splat_filter.ply_io import sh_dc_to_rgb
        n = 100
        rgb = sh_dc_to_rgb(np.zeros(n), np.zeros(n), np.zeros(n))
        assert rgb.shape == (n, 3)


class TestActualScale:
    """Test raw log-scale to actual scale conversion."""

    def test_exp_conversion(self):
        from msf.splat_filter.ply_io import actual_scale
        raw = np.array([0.0, 1.0, -1.0])
        expected = np.exp(raw)
        np.testing.assert_allclose(actual_scale(raw), expected)


class TestCOLMAPTextRoundTrip:
    """Test COLMAP points3D text format read/write round-trip."""

    def test_round_trip(self, tmp_path):
        from msf.colmap_filter.points3d_io import read_points3d, write_points3d

        # Create a small test file
        content = """# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 3
1 1.0 2.0 3.0 100 150 200 0.5 10 0 20 1
2 4.0 5.0 6.0 50 60 70 1.2 30 2
3 7.0 8.0 9.0 255 255 255 0.0 40 3 50 4 60 5
"""
        input_path = tmp_path / "points3D.txt"
        input_path.write_text(content)

        points = read_points3d(input_path)
        assert len(points) == 3

        # Verify point 1
        assert points[1]["xyz"][0] == pytest.approx(1.0)
        assert points[1]["rgb"][0] == 100
        assert points[1]["error"] == pytest.approx(0.5)
        assert len(points[1]["track"]) == 2

        # Verify point 3 has 3 track entries
        assert len(points[3]["track"]) == 3

        # Write and read back
        output_path = tmp_path / "points3D_out.txt"
        write_points3d(output_path, points)
        points2 = read_points3d(output_path)
        assert len(points2) == 3

        for pid in points:
            np.testing.assert_allclose(points[pid]["xyz"], points2[pid]["xyz"])
            np.testing.assert_array_equal(points[pid]["rgb"], points2[pid]["rgb"])
            assert points[pid]["error"] == pytest.approx(points2[pid]["error"])


class TestCOLMAPFilters:
    """Test individual COLMAP filter functions."""

    def test_track_length_filter(self):
        from msf.colmap_filter.filter import filter_by_track_length
        track_lengths = np.array([1, 2, 3, 4, 5, 10])
        mask = filter_by_track_length(track_lengths, min_length=3)
        expected = np.array([False, False, True, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_reproj_error_filter(self):
        from msf.colmap_filter.filter import filter_by_reproj_error
        errors = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        mask = filter_by_reproj_error(errors, max_error=1.0)
        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)


class TestSplatFilters:
    """Test splat filter functions with synthetic data."""

    def _make_structured_array(self, n):
        """Create a minimal structured array mimicking 3DGS PLY data."""
        dtype_fields = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        data = np.zeros(n, dtype=dtype_fields)
        return data

    def test_opacity_filter(self):
        from msf.splat_filter.filter import filter_by_opacity
        from msf.splat_filter.ply_io import inv_sigmoid
        data = self._make_structured_array(5)
        # Set opacity in logit space: sigmoid values of [0.01, 0.02, 0.05, 0.5, 0.99]
        targets = np.array([0.01, 0.02, 0.05, 0.5, 0.99])
        data["opacity"] = inv_sigmoid(targets).astype(np.float32)
        mask = filter_by_opacity(data, min_opacity=0.02)
        # 0.01 < 0.02 → remove; 0.02 >= 0.02 → keep
        assert not mask[0]  # 0.01 removed
        assert mask[2]      # 0.05 kept
        assert mask[3]      # 0.5 kept
        assert mask[4]      # 0.99 kept

    def test_elongation_filter(self):
        from msf.splat_filter.filter import filter_by_elongation
        data = self._make_structured_array(3)
        # Gaussian 0: roughly spherical (ratio ~1)
        data["scale_0"] = [0.0, 0.0, np.log(100.0)]
        data["scale_1"] = [0.0, 0.0, 0.0]
        data["scale_2"] = [0.0, 0.0, 0.0]
        # Gaussian 2: extremely elongated (ratio = 100)
        mask = filter_by_elongation(data, max_ratio=50.0)
        assert mask[0]       # spherical → keep
        assert mask[1]       # spherical → keep
        assert not mask[2]   # elongated → remove


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
