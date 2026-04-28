from __future__ import annotations

import json
import unittest

import src.bot_engine as bot_engine
from src.bot_engine import assess_question


class TmlHiraEngineTests(unittest.TestCase):
    def setUp(self):
        self._original_extract = bot_engine._extract_with_gemini
        bot_engine._extract_with_gemini = self._fake_extract_with_gemini

    def tearDown(self):
        bot_engine._extract_with_gemini = self._original_extract

    @staticmethod
    def _fake_extract_with_gemini(_components, user_text: str, _state: dict):
        text = user_text.strip().lower()
        if text == "welding on machine platform":
            return ({
                "activity": "welding on machine platform",
                "hazards": [{"hazard_type": "Fire / Explosion", "description": "Spark and flame risk from hot work."}],
            }, "")
        if text == "3 operators nearby":
            return ({"affected_people": "3 operators nearby", "people_score": 3}, "")
        if text == "direct":
            return ({"direct_indirect": "Direct"}, "")
        if text == "routine with sop":
            return ({"routine_nonroutine": "Routine"}, "")
        if text == "none":
            return ({"overriding_criteria": ["None"]}, "")
        if text == "hot work permit, loto, fire extinguisher and ppe":
            return ({"existing_controls": "hot work permit, LOTO, fire extinguisher and PPE"}, "")
        if text == "no major gap":
            return ({"gaps": "no major gap"}, "")
        if text == "fatal injury possible":
            return ({"injury_or_ill_health": "Injury", "harm_score": 5}, "")
        if text == "5":
            return ({"likelihood_score": 5}, "")
        if text == "3":
            return ({"scale_score": 3}, "")

        if text == "inspect electrical panel":
            return ({
                "activity": "inspect electrical panel",
                "hazards": [{"hazard_type": "Electrical", "description": "Electrical exposure during inspection."}],
            }, "")
        if text == "one operator":
            return ({"affected_people": "one operator", "people_score": 2}, "")
        if text == "lc legal concern":
            return ({"overriding_criteria": ["LC"]}, "")
        if text == "loto and electrical permit":
            return ({"existing_controls": "LOTO and electrical permit"}, "")
        if text == "first aid injury":
            return ({"injury_or_ill_health": "Injury", "harm_score": 3}, "")
        if text == "2":
            if _state.get("likelihood_score") is None:
                return ({"likelihood_score": 2}, "")
            return ({"scale_score": 2}, "")

        return ({}, "")

    def test_single_query_auto_completes_hira_draft(self):
        first = assess_question(None, "welding on machine platform", "{}")
        self.assertFalse(first["completed"])
        self.assertIn("How many people", first["answer"])

        result = assess_question(None, "3 employees", json.dumps(first["state"]))
        state = result["state"]

        self.assertTrue(result["completed"])
        self.assertIn("### TML HIRA Draft", result["answer"])
        self.assertIn("| Field | Details |", result["answer"])
        self.assertIn("Fire / Explosion", result["answer"])
        self.assertIn("| RPN | Likelihood x Severity", result["answer"])
        self.assertNotIn("deterministic fallback", result["answer"])
        self.assertEqual(state["activity"], "welding on machine platform")
        self.assertIsNotNone(state["likelihood_score"])
        self.assertIsNotNone(state["scale_score"])
        self.assertIn(state["harm_score"], {4, 5})
        self.assertEqual(state["people_score"], 3)

    def test_biw_hira_reference_controls_electrical_panel_scoring(self):
        first = assess_question(None, "inspect electrical panel", "{}")
        result = assess_question(None, "one operator", json.dumps(first["state"]))

        self.assertTrue(result["completed"])
        self.assertIn("Shock during cleaning activity", result["answer"])
        self.assertIn("| None | No over-riding criteria identified |", result["answer"])
        self.assertIn("| Risk level | Trivial", result["answer"])

    def test_rooftop_warehouse_query_uses_work_at_height_controls(self):
        first = assess_question(None, "i want to change the rooftop of the warehouse", "{}")
        result = assess_question(None, "4 contractors", json.dumps(first["state"]))

        self.assertTrue(result["completed"])
        self.assertIn("Gravity", result["answer"])
        self.assertIn("work-at-height permit", result["answer"])
        self.assertIn("| Risk level | Substantial", result["answer"])

    def test_biw_reference_marks_mfdc_gun_nonroutine(self):
        first = assess_question(None, "MFDC Gun", "{}")
        result = assess_question(None, "2 employees", json.dumps(first["state"]))

        self.assertTrue(result["completed"])
        self.assertIn("| Routine / Non-Routine | Non-Routine |", result["answer"])

    def test_generic_rooftop_change_does_not_use_unrelated_biw_rows(self):
        first = assess_question(None, "changing the roof tops of the warehouse", "{}")
        result = assess_question(None, "20", json.dumps(first["state"]))

        self.assertTrue(result["completed"])
        self.assertIn("| Routine / Non-Routine | Non-Routine |", result["answer"])
        self.assertNotIn("TML/DWD/BIW/MAINT/MANUAL GUNS", result["answer"])
        self.assertIn("Gravity", result["answer"])


if __name__ == "__main__":
    unittest.main()
