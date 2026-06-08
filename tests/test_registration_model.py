from __future__ import annotations

import unittest
from neurobench.manifests import load_json


class RegistrationModelTests(unittest.TestCase):
    def test_registration_model_validates_example(self):
        from neurobench.models.registration import RegistrationResult
        result = RegistrationResult.from_dict(load_json("examples/registration_result.example.json"))
        result.validate()
        self.assertEqual(result.to_dict()["transform"]["model"], "rigid")
