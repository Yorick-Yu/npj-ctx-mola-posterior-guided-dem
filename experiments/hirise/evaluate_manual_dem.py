"""Provisional metrics under explicitly estimated, non-reference-fitted grids."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parents[1] / "实验数据归档/npj_服务器完整备份_20260922/mars-npj-real-v1"
sys.path[:0] = [str(ROOT / "local-audit-deps"), str(ARCHIVE / "code/src")]
import csv
import argparse
import hashlib
import json
import numpy as np
from scipy.ndimage import map_coordinates, minimum_filter, uniform_filter, uniform_filter1d, correlate1d
from pyproj import CRS, Transformer
from paper2dem.detail_metrics import masked_gaussian_smooth, detail_field

INCOMING = ROOT / "incoming_dem_20260924"
OUT = INCOMING / "manual_alignment"
SNAP = ROOT / "remote_snapshot"
SEEDS = (20260808, 20260809, 20260810)
CORE = ("absolute_hirise_rmse_m", "mean_detail_rmse_m", "heldout_mola_rmse_m", "lowpass_rmse_m")


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def rmse(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else None


def sample(a, r, c):
    return map_coordinates(a, [r, c], order=1, mode="constant", cval=np.nan)


def restore(new, params):
    sr, sc, br, bc = params
    rr, cc = np.indices((504, 504), dtype=float)
    coords = [(rr-br)/sr, (cc-bc)/sc]
    finite = np.isfinite(new)
    values = map_coordinates(np.where(finite, new, 0), coords, order=1, mode="constant", cval=0.)
    support = map_coordinates(finite.astype(float), coords, order=1, mode="constant", cval=0.)
    valid = support >= 1.-1e-6
    return np.where(valid, values/np.maximum(support, 1e-12), np.nan), valid


def ground_spacing(provenance):
    mars = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
    inverse = Transformer.from_crs(CRS.from_wkt(provenance["output_crs"]), mars, always_xy=True)
    a,b,c,d,e,f = provenance["output_transform"]
    ys = f + e*(np.arange(504)+.5)
    xs = np.full(504, c+a*252)
    lon,lat = inverse.transform(xs,ys)
    lon_x,lat_x = inverse.transform(xs+a,ys)
    lon_y,lat_y = inverse.transform(xs,ys+e)
    geod = mars.get_geod()
    dx = np.asarray(geod.inv(lon,lat,lon_x,lat_x)[2])[:,None]
    dy = np.asarray(geod.inv(lon,lat,lon_y,lat_y)[2])[:,None]
    return dx,dy


def slope(field, valid, dx, dy):
    dr,dc = np.gradient(np.where(np.isfinite(field),field,0))
    values = np.degrees(np.arctan(np.hypot(dr/dy,dc/dx)))
    supported = minimum_filter(valid.astype(np.uint8),size=3,mode="constant",cval=0).astype(bool)
    return values,supported


def roughness(field, valid, nx, ny):
    # RMS residual after a least-squares local plane on a complete odd window.
    z = np.where(np.isfinite(field),field,0.).astype(float)
    mean = uniform_filter(z,size=(ny,nx),mode="constant",cval=0.)
    square = uniform_filter(z*z,size=(ny,nx),mode="constant",cval=0.)
    x = np.arange(nx)-(nx-1)/2
    y = np.arange(ny)-(ny-1)/2
    xz = uniform_filter1d(correlate1d(z,x/nx,axis=1,mode="constant",cval=0.),ny,axis=0,mode="constant",cval=0.)
    yz = uniform_filter1d(correlate1d(z,y/ny,axis=0,mode="constant",cval=0.),nx,axis=1,mode="constant",cval=0.)
    variance = square-mean*mean-xz*xz/np.mean(x*x)-yz*yz/np.mean(y*y)
    values = np.sqrt(np.maximum(variance,0))
    support = minimum_filter(valid.astype(np.uint8),size=(ny,nx),mode="constant",cval=0).astype(bool)
    return values,support


def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def formula_tests():
    rr,cc=np.indices((51,51),dtype=float)
    z=100+2*rr+3*cc;valid=np.ones(z.shape,bool)
    r,s=roughness(z,valid,9,9)
    assert np.max(r[s])<1e-4
    sl,s=slope(z,valid,np.full((51,1),20.),np.full((51,1),20.))
    assert np.max(np.abs(sl[s]-np.degrees(np.arctan(np.hypot(2,3)/20))))<1e-10
    values,s=roughness(z+5*np.sin(cc),valid,9,9)
    window=(z+5*np.sin(cc))[21:30,21:30];yy,xx=np.indices((9,9));design=np.column_stack([np.ones(81),xx.ravel(),yy.ravel()])
    fit=design@np.linalg.lstsq(design,window.ravel(),rcond=None)[0]
    assert abs(values[25,25]-rmse(window.ravel()-fit))<1e-8


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--declared-input-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    out=args.output_dir or OUT
    out.mkdir(parents=True,exist_ok=True)
    formula_tests()
    grid=json.loads((OUT/"grid-diagnostics.json").read_text()) if not args.declared_input_dir else None
    new_directory=args.declared_input_dir or INCOMING/"predictions"
    protocol=dict(status="diagnostic_only_not_reproduced_final_results", references_used_for_grid_fit=False,
                  vertical_adjustment=False, invalid_pixel_filling=False, masks="same valid pixels/points across all methods within each scenario",
                  slopes="central differences on archived grid with row-wise planetary ground distances; units degrees",
                  roughness="local least-squares plane residual RMS; nominal ground spans 80,160,320 m, nearest odd window; complete support only",
                  uncertainty="Provisional geometric restoration does not recover missing generation provenance.",
                  scenarios=["declared_full_extent"] if args.declared_input_dir else ["best_integer_crop","full_center_extent","estimated_shared"],
                  new_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in new_directory.glob("*.npz")})
    if args.declared_input_dir:
        protocol.update(grid_source="Coordinates assigned by user-supplied resampling script; not recovered original georeferencing",
                        alignment_fitted=False,source_rasters_resampled_before_delivery=True)
    (out/"evaluation-protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    all_rows=[];summaries=[];context=[]
    for region in ("candor","hellespontus"):
        rdir=SNAP/"cases"/region
        ref=read_npz(rdir/"reference.npz");inp=read_npz(rdir/"inference.npz")
        provenance=json.loads((rdir/"provenance.json").read_text())
        dx,dy=ground_spacing(provenance)
        if args.declared_input_dir:
            scenarios={"declared_full_extent":[1.,1.,0.,0.]}
        else:
            fixed=grid["regions"][region]["fixed_candidates"][0]
            scenarios={"best_integer_crop":fixed["params"],"full_center_extent":[503/500,503/500,0.,0.],
                       "estimated_shared":grid["regions"][region]["estimated_shared_mapping"]["params"]}
        for scenario,params in scenarios.items():
            fields=[("coarse_CTX",0,0,inp["z0"])]
            new_fields=[]
            for seed in SEEDS:
                new=read_npz(new_directory/f"{region.title()}_{seed}.npz")["dem"]
                if args.declared_input_dir:
                    assert new.shape==(504,504)
                    field=new.astype(float);valid=np.isfinite(field)
                else:
                    field,valid=restore(new,params)
                fields.append(("incoming_full",seed,20,field));new_fields.append(field)
                for method,budget,label in [("full",0,"full"),("full",20,"archived_full"),("plain_unet",20,"plain_unet")]:
                    f=read_npz(SNAP/"predictions"/region/f"{method}-{seed}-k{budget}-final.npz")["mean"]
                    fields.append((label,seed,budget,f))
            fields.append(("GP",0,20,read_npz(SNAP/"predictions"/region/"GP-0-k20-final.npz")["mean"]))
            availability=np.logical_and.reduce([np.isfinite(f) for f in new_fields])
            common=ref["valid"]&availability
            heldout=sample(availability.astype(float),ref["heldout_row"],ref["heldout_column"])>=1.-1e-6
            for new in new_fields:
                heldout &= np.isfinite(sample(new,ref["heldout_row"],ref["heldout_column"]))
            ref_slope,smask=slope(ref["dem"],common,dx,dy)
            rough_refs={};window_info=[]
            for span in (80,160,320):
                nx=2*max(1,int(round(span/(2*float(dx[252,0])))))+1
                ny=2*max(1,int(round(span/(2*float(dy[252,0])))))+1
                rough_refs[span]=(*roughness(ref["dem"],common,nx,ny),nx,ny)
                window_info.append(dict(nominal_span_m=span,columns=nx,rows=ny,center_east_span_m=(nx-1)*float(dx[252,0]),center_north_span_m=(ny-1)*float(dy[252,0])))
            context.append(dict(region=region,scenario=scenario,params=params,valid_reference_pixels=int(ref["valid"].sum()),
                                common_reference_pixels=int(common.sum()),heldout_total=int(len(heldout)),heldout_used=int(heldout.sum()),
                                east_ground_spacing_m=float(dx[252,0]),north_ground_spacing_m=float(dy[252,0]),roughness_windows=window_info))
            for method,seed,budget,field in fields:
                row=dict(region=region,scenario=scenario,method=method,seed=seed,budget=budget,
                         reference_pixels=int(common.sum()),heldout_points=int(heldout.sum()),
                         absolute_hirise_rmse_m=rmse((field-ref["dem"])[common]))
                details=[]
                for scale in (40,80,160):
                    fp,fs=detail_field(field,common,pixel_size_m=20,scale_m=scale)
                    rp,rs=detail_field(ref["dem"],common,pixel_size_m=20,scale_m=scale)
                    details.append(rmse((fp-rp)[fs&rs]))
                row["mean_detail_rmse_m"]=float(np.mean(details))
                fl,fs=masked_gaussian_smooth(field,common,sigma_pixels=16)
                rl,rs=masked_gaussian_smooth(ref["dem"],common,sigma_pixels=16)
                row["lowpass_rmse_m"]=rmse((fl-rl)[fs&rs]);row["lowpass_pixels"]=int((fs&rs).sum())
                he=sample(field,ref["heldout_row"],ref["heldout_column"])-ref["heldout_elevation"]
                row["heldout_mola_rmse_m"]=rmse(he[heldout])
                sl,ss=slope(field,common,dx,dy);ss &= smask
                row["slope_mae_deg"]=float(np.mean(np.abs(sl[ss]-ref_slope[ss])))
                row["slope_rmse_deg"]=rmse(sl[ss]-ref_slope[ss]);row["slope_pixels"]=int(ss.sum())
                row["slope_mean_deg"]=float(np.mean(sl[ss]));row["reference_slope_mean_deg"]=float(np.mean(ref_slope[ss]))
                for span,(rv,rs,nx,ny) in rough_refs.items():
                    fv,fs=roughness(field,common,nx,ny);support=fs&rs
                    row[f"roughness_{span}m_mae_m"]=float(np.mean(np.abs(fv[support]-rv[support])))
                    row[f"roughness_{span}m_rmse_m"]=rmse(fv[support]-rv[support])
                    row[f"roughness_{span}m_mean_m"]=float(np.mean(fv[support]))
                    row[f"reference_roughness_{span}m_mean_m"]=float(np.mean(rv[support]))
                    row[f"roughness_{span}m_pixels"]=int(support.sum())
                all_rows.append(row)
            for method,budget in [("coarse_CTX",0),("full",0),("incoming_full",20),("archived_full",20),("plain_unet",20),("GP",20)]:
                subset=[r for r in all_rows if (r["region"],r["scenario"],r["method"],r["budget"])==(region,scenario,method,budget)]
                summary=dict(region=region,scenario=scenario,method=method,budget=budget,seed_count=len(subset))
                for key in subset[0]:
                    if key not in ("region","scenario","method","seed","budget"):
                        summary[key]=float(np.mean([r[key] for r in subset]))
                summaries.append(summary)
            print(json.dumps(dict(region=region,scenario=scenario,common_pixels=int(common.sum()),heldout_used=int(heldout.sum()),
                                  incoming={key:summaries[-4][key] for key in CORE})),flush=True)
    write_csv(out/"diagnostic-per-seed.csv",all_rows);write_csv(out/"diagnostic-summary.csv",summaries)
    (out/"evaluation-context.json").write_text(json.dumps(context,indent=2)+"\n")
    print("COMPLETE: provisional core metrics and slope/roughness diagnostics; no final result table changed.")


if __name__=="__main__":
    main()
