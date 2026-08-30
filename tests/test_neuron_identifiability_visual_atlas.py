from neurobench.experiments.neuron_identifiability.visual_statistical_atlas import select_representatives


def test_representative_selection_is_objective_and_distinct():
    rows=[]
    for i in range(5): rows.append({"observation_site_id":f"r{i}","occurrences":4,"confirmed_only":True,"composite_score":i,"recovery_score":i/4,"measurement_score":i/4,"typical_distance":abs(i-2)})
    result=select_representatives(rows)
    assert result["hero"]["observation_site_id"]=="r4"
    assert result["hard_case"]["observation_site_id"]=="r0"
    assert result["typical"]["observation_site_id"]=="r2"
