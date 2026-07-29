import unittest

from streamlit.testing.v1 import AppTest


class ApplicationSmokeTests(unittest.TestCase):
    def test_default_simulation_completes_without_streamlit_exceptions(self):
        app = AppTest.from_file(
            "Streamlit_App/app/main.py",
            default_timeout=120,
        )
        app.run()
        self.assertEqual([], list(app.exception))
        self.assertIn("▶️ Run Simulation", [button.label for button in app.button])

        run_button = next(
            button for button in app.button if button.label == "▶️ Run Simulation"
        )
        run_button.click().run(timeout=120)
        self.assertEqual([], list(app.exception))
        self.assertTrue(app.session_state["run_history"])
        self.assertIn("latest_result_full", app.session_state)
        self.assertGreater(len(app.get("plotly_chart")), 0)


if __name__ == "__main__":
    unittest.main()
