from neurobench.experiments.neuron_identifiability.ica_morphology_ablation import _stratum,select_queue
def test_strata_prioritize_artifact_overlap_and_weak_signal():
 base={"disposition":"confirmed_neuron","morphology":"localized_center","context":"isolated","observation_id":"x"}
 assert _stratum(base)=="localized_center"
 assert _stratum({**base,"disposition":"artifact"})=="artifact_control"
 assert _stratum({**base,"morphology":"overlap"})=="overlap"
 assert _stratum({**base,"context":"weak_signal"})=="weak_signal"
def test_queue_is_deterministic_and_bounded():
 rows=[]
 for i in range(8):rows.append({"observation_id":str(i),"disposition":"confirmed_neuron","morphology":"localized_center","context":"isolated"})
 assert select_queue(rows,2)==select_queue(rows,2)
 assert len(select_queue(rows,2))==2
