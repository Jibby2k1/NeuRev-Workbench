from neurobench.experiments.neuron_identifiability.cross_neural_class_extensions import _site_classes


def test_site_classes_matches_nearest_detection_within_radius():
    sites=[{"site_id":"a","x_px":"0","y_px":"0"}]
    detections=[{"burst_id":"1","x_px":"3","y_px":"4","class_id":"2"},{"burst_id":"1","x_px":"1","y_px":"1","class_id":"3"}]
    assert _site_classes(sites,detections)=={("a",1):3}
