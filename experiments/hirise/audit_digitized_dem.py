"""Independently test whether delivered DEMs are inverse-color reconstructions."""

import sys
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"local-audit-deps"))
import hashlib
import json
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

SOURCE=Path(os.environ.get("DEM_NPZ_ROOT", "data/DEM_npz"))
IMAGE=Path(os.environ.get("DEM_PANEL_IMAGE", "data/dem-panel.png"))
OUT=ROOT/"incoming_dem_digitized_20260924"
PANELS=((109.,647.),(754.5,1292.5),(1401.,1939.))
SPECS={
    "Candor":dict(panel_y=(79.5,617.5),bar_y=(43,655),ticks_y=(74,172,272,370,470,568),ticks_z=(1300,1200,1100,1000,900,800)),
    "Hellespontus":dict(panel_y=(818.5,1357.5),bar_y=(782,1394),ticks_y=(853,966,1081,1194,1307),ticks_z=(-100,-200,-300,-400,-500)),
}


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def from_colors(image,spec,panel):
    by=np.arange(*spec["bar_y"])
    colors=np.median(image[by,1992:2012],axis=1)
    slope,intercept=np.polyfit(spec["ticks_y"],spec["ticks_z"],1)
    heights=slope*by+intercept
    left,right=panel;top,bottom=spec["panel_y"]
    x0,x1=int(np.ceil(left+2)),int(np.floor(right-2))
    y0,y1=int(np.ceil(top+2)),int(np.floor(bottom-2))
    rgb=image[y0:y1+1,x0:x1+1]
    distance,index=cKDTree(colors).query(rgb.reshape(-1,3))
    z=heights[index].reshape(rgb.shape[:2])
    missing=(distance.reshape(z.shape)>18)|(rgb.min(axis=2)>210)
    centers=(np.arange(504)+.5)/504
    coordinates=np.array(np.meshgrid(top+centers*(bottom-top)-y0,left+centers*(right-left)-x0,indexing="ij"))
    weight=map_coordinates((~missing).astype(float),coordinates,order=1,mode="nearest")
    weighted=map_coordinates(np.where(missing,0,z),coordinates,order=1,mode="nearest")
    keep=~map_coordinates(missing.astype(np.uint8),coordinates,order=0,mode="nearest").astype(bool)
    result=np.full((504,504),np.nan,np.float32)
    np.divide(weighted,weight,out=result,where=keep&(weight>0))
    return result,dict(colorbar_height_m_per_image_pixel=float(abs(slope)),palette_rows=int(len(colors)),
                       unique_palette_rgb=int(len(np.unique(colors,axis=0))))


def main():
    (OUT/"files").mkdir(parents=True,exist_ok=True)
    image=np.asarray(Image.open(IMAGE).convert("RGB"),dtype=float)
    assert image.shape==(1444,2154,3)
    files=sorted(SOURCE.glob("*.npz"))+[SOURCE/"restore_dem.py",SOURCE/"update_dem_coordinates.py"]
    source_hashes={p.name:sha(p) for p in files}
    for p in files:
        dest=OUT/"files"/p.name
        if dest.exists():assert sha(dest)==source_hashes[p.name]
        else:shutil.copy2(p,dest)
        assert sha(dest)==source_hashes[p.name]
    records=[]
    for region,spec in SPECS.items():
        for seed,panel in zip((20260808,20260809,20260810),PANELS):
            path=SOURCE/f"{region}_{seed}.npz"
            with np.load(path,allow_pickle=False) as z:actual=z["dem"]
            expected,details=from_colors(image,spec,panel)
            joint=np.isfinite(expected)&np.isfinite(actual)
            records.append(dict(region=region,seed_label=seed,file=path.name,sha256=sha(path),
                                exact_array_match_to_colorbar_reconstruction=bool(np.array_equal(actual,expected,equal_nan=True)),
                                max_absolute_difference_m=float(np.max(np.abs(actual[joint]-expected[joint]))),
                                matching_nan_mask=bool(np.array_equal(np.isnan(actual),np.isnan(expected))),
                                nan_pixels=int(np.isnan(actual).sum()),**details))
    exact=all(r["exact_array_match_to_colorbar_reconstruction"] for r in records)
    report=dict(status="image_digitization_confirmed" if exact else "image_digitization_not_exactly_confirmed",
                source_image=str(IMAGE),source_image_sha256=sha(IMAGE),source_image_pixels=[2154,1444],
                source_code_executed=False,all_six_exactly_reconstructed=exact,
                conclusion="These are inverse-color estimates from a rendered image, not recovered model output arrays.",
                coordinate_origin="Old bounds assigned by update_dem_coordinates.py; original new-run georeferencing remains unverified.",
                permitted_scope="Approximate figure inspection, with digitization uncertainty; not validation of raw-model slope/roughness or exact RMSE.",
                original_and_reported_results_untouched=True,files=records,source_hashes=source_hashes)
    (OUT/"digitization-audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# 当前六张DEM的来源核验", "", "## 核验结论", "",
           "六个DEM数组均可由用户提供的彩色六联图，经颜色条匹配和插值逐值重建；与当前NPZ完全一致，最大绝对差为0米，NaN掩膜也一致。" if exact else "尚未确认全部文件可由彩图逐值重建，见JSON逐文件检查。", "",
           "因此，这些文件是彩图数字化估计，不是恢复出的模型原始预测。此处未执行会覆盖文件的用户脚本；用单独核验实现计算，并保留了当前文件的只读副本和SHA256。", "",
           "## 具体证据", "", "| 区域 | 种子标签 | 与彩图反演逐值一致 | 最大绝对差（米） | NaN像元 |", "|---|---:|---|---:|---:|"]
    for r in records:
        lines.append(f'| {r["region"]} | {r["seed_label"]} | {r["exact_array_match_to_colorbar_reconstruction"]} | {r["max_absolute_difference_m"]:.6f} | {r["nan_pixels"]} |')
    lines += ["", "## 代码实际做了什么", "",
              "1. restore_dem.py 从截图右侧颜色条读取RGB与高程对应关系，使用最近颜色匹配把每个面板的RGB像素反推为高程，再插值为504×504。",
              "2. 白色或颜色不匹配的像素被标为缺失值；这些缺失不等价于经过核验的原始DEM有效掩膜。",
              "3. update_dem_coordinates.py 赋予旧网格的坐标范围，没有提供新服务器原始输出的地理参考记录。",
              "4. 文件名中的种子编号是面板标签，不能代替模型权重、运行日志或逐种子原始预测的来源证明。", "",
              "## 对论文分析的影响", "",
              "可以按这些数组描述或近似复现彩图，但不能把它们标为实测或模型直接输出的高程来验证最终RMSE、坡度或粗糙度。颜色量化、渲染、截图和再次插值的误差未标定，地形导数可能受到影响。",
              "先前对已补坐标版本的物理量计算属于假设网格下的诊断，不应转写成正式新实验结果。新汇总表也仍未由可追溯原始预测复算确认。",
              "原始旧实验归档及用户提供的指标表均未修改；当前彩图反演版本单独保存，不覆盖旧模型输出。",
              "正式物理指标验证应使用可追溯的原始预测：已核验的旧504×504预测可以计算并明确标为旧实验；若采用新版，应找回原始输出或用固定代码、数据和权重重新推理。", "",
              "机器可读核验见digitization-audit.json；文件副本及用户脚本见files/。"]
    (OUT/"彩图反演来源核验.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    for p in files:assert sha(p)==source_hashes[p.name]
    print(json.dumps(dict(status=report["status"],all_six_exactly_reconstructed=exact,
                          per_file_max_abs_difference=[r["max_absolute_difference_m"] for r in records],
                          report=str(OUT/"彩图反演来源核验.md")),ensure_ascii=False))


if __name__=="__main__":main()
