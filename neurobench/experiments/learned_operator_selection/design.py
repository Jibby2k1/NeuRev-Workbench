"""Frozen nested Sobol designs and deterministic parameter transforms."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import qmc

FAMILY_DIMENSIONS={"spatial":5,"temporal":5,"separable_spatiotemporal":7}
FAMILY_SEED_OFFSETS={"spatial":0,"temporal":1,"separable_spatiotemporal":2}

def _linear(value:float,low:float,high:float)->float: return low+value*(high-low)
def _log(value:float,low:float,high:float)->float: return float(math.exp(math.log(low)+value*(math.log(high)-math.log(low))))
def _odd(value:float,low:int,high:int)->int:
    choices=np.arange(low if low%2 else low+1,high+1,2); return int(choices[min(int(value*len(choices)),len(choices)-1)])

def sobol_master(family:str, *, seed:int=20260819, size:int=64) -> np.ndarray:
    if family not in FAMILY_DIMENSIONS or size<1 or size&(size-1): raise ValueError("family must be known and size a power of two")
    return qmc.Sobol(FAMILY_DIMENSIONS[family],scramble=True,seed=seed+FAMILY_SEED_OFFSETS[family]).random_base2(int(math.log2(size)))

def map_point(family:str,index:int,point:np.ndarray)->dict[str,Any]:
    u=np.asarray(point,dtype=float)
    common={"family":family,"design_index":int(index),"shrinkage":_log(float(u[-3]),1e-4,.5),"eigen_floor_ratio":_log(float(u[-2]),1e-6,1e-2),"raw_blend":_linear(float(u[-1]),.1,1.0)}
    if family=="spatial": return {**common,"spatial_width_px":_odd(u[0],3,21),"spatial_exponent":float(u[1])}
    if family=="temporal": return {**common,"temporal_width_frames":_odd(u[0],3,33),"temporal_exponent":float(u[1])}
    if family=="separable_spatiotemporal": return {**common,"spatial_width_px":_odd(u[0],3,15),"temporal_width_frames":_odd(u[1],3,25),"spatial_exponent":float(u[2]),"temporal_exponent":float(u[3])}
    raise ValueError(family)

def unique_prefix(family:str,count:int,*,seed:int=20260819,size:int=64)->list[dict[str,Any]]:
    if count not in {8,16,32,64}: raise ValueError("prefix must be 8,16,32,64")
    rows=[]; seen=set()
    for index,point in enumerate(sobol_master(family,seed=seed,size=size)):
        row=map_point(family,index,point); key=json.dumps({k:v for k,v in row.items() if k!="design_index"},sort_keys=True)
        if key in seen: continue
        seen.add(key); rows.append(row)
        if len(rows)==count: return rows
    raise RuntimeError(f"master design has only {len(rows)} unique {family} specifications")

def design_hash(rows:list[dict[str,Any]])->str:
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def lhs_confirmation(family:str,*,seed:int=20260829,size:int=16)->np.ndarray:
    if family not in FAMILY_DIMENSIONS: raise ValueError(family)
    return qmc.LatinHypercube(FAMILY_DIMENSIONS[family],scramble=True,seed=seed+FAMILY_SEED_OFFSETS[family]).random(size)

def write_master_designs(directory: str|Path, *, seed:int=20260819, size:int=64) -> dict[str,Any]:
    root=Path(directory); root.mkdir(parents=True,exist_ok=True); result={}
    for family in FAMILY_DIMENSIONS:
        rows=[map_point(family,index,point) for index,point in enumerate(sobol_master(family,seed=seed,size=size))]
        path=root/f"{family}_sobol_master.tsv"
        if path.exists():
            raise FileExistsError(f"master design already exists: {path}")
        fields=sorted({key for row in rows for key in row}); temporary=path.with_suffix(".partial")
        with temporary.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t"); writer.writeheader(); writer.writerows(rows)
        temporary.replace(path); result[family]={"path":str(path),"rows":len(rows),"design_hash":design_hash(rows)}
    return result
