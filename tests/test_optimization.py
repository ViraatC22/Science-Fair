import unittest

import numpy as np

from backend.nn import DEFAULT_INPUT_DIM, MLP, train_surrogate
from backend.opt import ORDER, sample_params, to_vector, validate_params


class OptimizationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.params = {
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

    def test_model_and_parameter_vector_use_same_feature_count(self):
        self.assertEqual(18, len(ORDER))
        self.assertEqual(len(ORDER), DEFAULT_INPUT_DIM)
        self.assertEqual(len(ORDER), MLP().l1.in_features)
        self.assertEqual((len(ORDER),), to_vector(self.params).shape)

    def test_validation_rejects_missing_nonfinite_and_fractional_integer_values(self):
        cases = []
        missing = dict(self.params)
        missing.pop("ion_ca")
        cases.append(missing)
        cases.append({**self.params, "light_lumens": np.nan})
        cases.append({**self.params, "pillar_count": 4.5})

        for params in cases:
            with self.subTest(params=params):
                valid, _ = validate_params(params)
                self.assertFalse(valid)
                with self.assertRaises(ValueError):
                    to_vector(params)

    def test_validation_rejects_scaffolds_larger_than_simulation_domain(self):
        valid, message = validate_params(
            {
                **self.params,
                "pillar_count": 20,
                "pillar_size_mm": 50.0,
                "channel_width_mm": 30.0,
            }
        )
        self.assertFalse(valid)
        self.assertIn("exceeds", message)

    def test_sampled_parameters_are_always_valid(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            params = sample_params(rng)
            self.assertTrue(validate_params(params)[0])
            self.assertTrue(np.isfinite(to_vector(params)).all())

    def test_training_rejects_wrong_feature_shape(self):
        with self.assertRaisesRegex(ValueError, "18"):
            train_surrogate(
                np.zeros((5, 17), dtype=np.float32),
                np.zeros((5, 1), dtype=np.float32),
                epochs=1,
            )


if __name__ == "__main__":
    unittest.main()
