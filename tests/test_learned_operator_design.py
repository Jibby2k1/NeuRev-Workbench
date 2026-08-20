import numpy as np
from neurobench.experiments.learned_operator_selection.design import design_hash,lhs_confirmation,sobol_master,unique_prefix,write_master_designs

def test_sobol_prefixes_are_reproducible_and_nested() -> None:
    a=sobol_master("spatial"); b=sobol_master("spatial"); np.testing.assert_array_equal(a,b)
    p8=unique_prefix("spatial",8); p16=unique_prefix("spatial",16)
    assert p16[:8]==p8 and design_hash(p8)==design_hash(unique_prefix("spatial",8))
    assert all(row["spatial_width_px"]%2==1 and 3<=row["spatial_width_px"]<=21 for row in p16)

def test_family_designs_and_lhs_are_independent() -> None:
    assert sobol_master("spatial").shape==(64,5); assert sobol_master("separable_spatiotemporal").shape==(64,7)
    lhs=lhs_confirmation("temporal"); assert lhs.shape==(16,5) and np.all((lhs>=0)&(lhs<=1))
    assert not np.array_equal(lhs,sobol_master("temporal")[:16])

def test_master_designs_are_written_once_before_evaluation(tmp_path) -> None:
    result=write_master_designs(tmp_path); assert set(result)=={"spatial","temporal","separable_spatiotemporal"}
    assert all(item["rows"]==64 for item in result.values())
    try: write_master_designs(tmp_path)
    except FileExistsError: pass
    else: raise AssertionError("master design overwrite was allowed")
