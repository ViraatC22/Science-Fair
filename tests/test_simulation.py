import unittest

import numpy as np

from Streamlit_App.simulation.physics import (
    compute_metrics,
    generate_micropillar_geometry,
    run_simulation_logic,
)


PARAMS = {
    "model_type": "3D Structured (Channel Flow)",
    "pillar_count": 4,
    "pillar_size_mm": 30.0,
    "channel_width_mm": 15.0,
    "channel_node_size_mm": 5.0,
    "scaffold_stiffness_kPa": 25.0,
    "elasticity": 0.5,
    "scaffold_density_g_cm3": 1.2,
    "initial_mass_g": 0.5,
    "media_depth_mm": 2.0,
    "replenish_freq_hr": 24,
    "dmem_glucose": 25.0,
    "dmem_glutamine": 45.0,
    "dmem_pyruvate": 1.0,
    "ion_na": 154.0,
    "ion_k": 5.4,
    "ion_cl": 140.0,
    "ion_ca": 1.8,
    "light_lumens": 0.0,
}


class SimulationTests(unittest.TestCase):
    def test_seed_reproduces_growth_field_and_metrics(self):
        first = run_simulation_logic(
            PARAMS,
            PARAMS["model_type"],
            rng=np.random.default_rng(99),
        )
        second = run_simulation_logic(
            PARAMS,
            PARAMS["model_type"],
            rng=np.random.default_rng(99),
        )
        np.testing.assert_array_equal(first["growth_data"], second["growth_data"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_all_models_return_finite_core_metrics(self):
        for model_type in (
            "2.5D Surface (Pillar Tops)",
            "3D Porous (Channel Diffusion)",
            "3D Structured (Channel Flow)",
        ):
            with self.subTest(model_type=model_type):
                metrics = compute_metrics(
                    PARAMS,
                    model_type,
                    rng=np.random.default_rng(12),
                )
                self.assertGreater(metrics["avg_growth_rate"], 0)
                self.assertGreater(metrics["total_network_length"], 0)
                self.assertTrue(
                    np.isfinite(
                        [metrics["avg_growth_rate"], metrics["total_network_length"]]
                    ).all()
                )

    def test_invalid_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            compute_metrics(PARAMS, "not-a-model")
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            run_simulation_logic(PARAMS, "not-a-model")

    def test_impossible_geometry_is_rejected_instead_of_silently_clipped(self):
        invalid = {
            **PARAMS,
            "pillar_count": 20,
            "pillar_size_mm": 50.0,
            "channel_width_mm": 30.0,
        }
        with self.assertRaisesRegex(ValueError, "exceeds"):
            generate_micropillar_geometry(invalid)


if __name__ == "__main__":
    unittest.main()
