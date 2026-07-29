import unittest

import numpy as np

from Streamlit_App.simulation.export_utils import (
    calculate_voxel_size,
    export_to_bundle,
    generate_heightmap_mesh,
    generate_improved_mesh,
)


class ExportTests(unittest.TestCase):
    def test_heightmap_export_is_watertight(self):
        mesh = generate_heightmap_mesh(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        )
        self.assertTrue(mesh.is_watertight)
        bundle = export_to_bundle(mesh, filename_base="test-scaffold")
        self.assertIn(b"test-scaffold.stl", bundle)
        self.assertIn(b"ONSHAPE_IMPORT_SETTINGS.txt", bundle)

    def test_voxel_export_is_watertight(self):
        volume = np.zeros((5, 5, 5), dtype=float)
        volume[1:4, 1:4, 1:4] = 1.0
        mesh = generate_improved_mesh(volume)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(len(mesh.faces), 0)

    def test_invalid_mesh_inputs_have_clear_errors(self):
        invalid_heightmaps = (np.empty((0, 0)), np.ones((1, 1)))
        for heightmap in invalid_heightmaps:
            with self.subTest(shape=heightmap.shape):
                with self.assertRaisesRegex(ValueError, "at least 2x2"):
                    generate_heightmap_mesh(heightmap)

        with self.assertRaisesRegex(ValueError, "both solid and empty"):
            generate_improved_mesh(np.ones((4, 4, 4)))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            calculate_voxel_size({}, grid_size=0)


if __name__ == "__main__":
    unittest.main()
