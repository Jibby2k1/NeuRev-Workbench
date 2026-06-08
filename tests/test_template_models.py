from __future__ import annotations

import unittest
from neurobench.manifests import load_json


class TemplateModelTests(unittest.TestCase):
    def test_template_model_validates_example(self):
        from neurobench.models.template import TemplateSpec
        spec = TemplateSpec.from_dict(load_json("examples/template_spec.example.json"))
        spec.validate()
        self.assertEqual(spec.to_dict()["template_id"], "template_from_1_neutral_v1")
