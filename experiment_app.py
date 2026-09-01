import os
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# 1. 기본 설정
# =========================================================
DATA_TYPE_OPTIONS = {
    "일반 실험 / 반복실험": {
        "mode": "general",
        "desc": "변수 간 경향, 반복 측정, 회귀 관계를 확인하는 일반 실험 데이터",
    },
    "재료 시험 (인장·압축·굽힘 등)": {
        "mode": "material",
        "desc": "응력-변형률, 하중-변위 등 곡선 자체와 물성값이 중요한 데이터",
    },
    "분광 분석 (FT-IR / UV-Vis)": {
        "mode": "spectroscopy",
        "desc": "파장/파수에 따른 흡광도·투과도 등 스펙트럼 데이터",
    },
    "열분석 (TGA / DSC)": {
        "mode": "thermal",
        "desc": "온도에 따른 질량 변화 또는 열유속 변화 데이터",
    },
    "캘리브레이션 / 보정": {
        "mode": "calibration",
        "desc": "표준값과 측정 신호 사이의 선형 보정 관계를 확인하는 데이터",
    },
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
MAX_ROWS = 50_000
MAX_MB = 10
ENCODINGS = ["utf-8", "cp949", "euc-kr", "utf-8-sig"]


def register_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for path in candidates:
        try:
            pdfmetrics.registerFont(TTFont("Korean", path))
            return "Korean"
        except Exception:
            continue
    return "Helvetica"


FONT_NAME = register_korean_font()


# =========================================================
# 2. 공통 유틸리티
# =========================================================
def load_csv(uploaded_file):
    for enc in ENCODINGS:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            df.columns = df.columns.astype(str).str.strip()
            return df, enc
        except Exception:
            continue
    return None, None


def clean_xy(df, x_col, y_col):
    data = df[[x_col, y_col]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    return data


def axis_label(column, unit):
    unit = (unit or "").strip()
    return f"{column} ({unit})" if unit else column


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def file_signature(loaded, x_col, y_col, data_type_key, extra_settings):
    """파일/축/분석설정이 바뀌면 이전 AI 해석을 폐기하기 위한 서명."""
    h = hashlib.sha256()
    h.update(str(x_col).encode())
    h.update(str(y_col).encode())
    h.update(str(data_type_key).encode())
    h.update(repr(extra_settings).encode())
    for item in loaded:
        h.update(item["name"].encode(errors="ignore"))
        # 전체 파일 bytes를 다시 읽지 않고 dataframe 내용으로 안정적인 서명 생성
        try:
            hashed = pd.util.hash_pandas_object(item["df"], index=True).values.tobytes()
            h.update(hashed)
        except Exception:
            h.update(str(item["df"].shape).encode())
    return h.hexdigest()


def detect_outliers_zscore(data, x_col, y_col, threshold):
    """일반 통계 데이터에서만 사용하는 단순 Z-score 이상치 탐지."""
    if len(data) < 4:
        return pd.Series(False, index=data.index)
    values = data[[x_col, y_col]].astype(float).values
    z = np.abs(stats.zscore(values, axis=0, nan_policy="omit"))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series((z > threshold).any(axis=1), index=data.index)


def repeated_summary(data, x_col, y_col):
    summary = (
        data.groupby(x_col)[y_col]
        .agg(n="count", mean="mean", sd="std", sem="sem")
        .reset_index()
    )
    return summary.sort_values(x_col)


def has_replicates(data, x_col):
    return data[x_col].duplicated(keep=False).any()


# =========================================================
# 3. 일반 실험 회귀 분석
# =========================================================
def analyze_general(df, x_col, y_col, remove_outliers=False, zscore_threshold=3.0):
    raw = clean_xy(df, x_col, y_col)
    if len(raw) < 3:
        return None

    outlier_mask = detect_outliers_zscore(raw, x_col, y_col, zscore_threshold)
    outlier_count = int(outlier_mask.sum())
    data = raw.loc[~outlier_mask].copy() if remove_outliers else raw.copy()

    if len(data) < 3 or data[x_col].nunique() < 2:
        return None

    X = data[[x_col]].values
    y = data[y_col].values.astype(float)
    n = len(data)

    # 복잡한 모델은 데이터가 충분할 때만 후보로 허용한다.
    degrees = [1]
    if n >= 6 and data[x_col].nunique() >= 4:
        degrees.append(2)
    if n >= 8 and data[x_col].nunique() >= 5:
        degrees.append(3)

    model_results = {}

    # 검증 fold에 최소 2개 정도가 들어가도록 n_splits를 제한한다.
    can_cv = n >= 5
    cv = None
    if can_cv:
        n_splits = min(5, max(2, n // 2))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
        model.fit(X, y)
        pred = model.predict(X)
        train_r2 = float(r2_score(y, pred))
        train_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

        # 조정 R²: 모델 복잡도 증가를 어느 정도 패널티로 반영
        p = degree
        if n > p + 1:
            adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = np.nan

        cv_rmse = np.nan
        cv_rmse_std = np.nan
        if cv is not None:
            try:
                neg_mse = cross_val_score(
                    model, X, y, cv=cv, scoring="neg_mean_squared_error"
                )
                fold_rmse = np.sqrt(np.maximum(-neg_mse, 0))
                cv_rmse = float(np.mean(fold_rmse))
                cv_rmse_std = float(np.std(fold_rmse))
            except Exception:
                pass

        model_results[degree] = {
            "model": model,
            "train_r2": train_r2,
            "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
            "train_rmse": train_rmse,
            "cv_rmse": cv_rmse,
            "cv_rmse_std": cv_rmse_std,
            "pred": pred,
        }

    # 최적 모델은 '훈련 R² 최대'가 아니라 CV RMSE 최소를 사용한다.
    # 단, 최소 RMSE의 5% 이내라면 더 단순한 모델을 선택한다.
    valid_cv = {
        d: r["cv_rmse"] for d, r in model_results.items()
        if np.isfinite(r["cv_rmse"])
    }
    if valid_cv:
        min_rmse = min(valid_cv.values())
        # CV 오차가 0에 매우 가까운 완전/준완전 적합에서도 부동소수점 차이 때문에
        # 불필요하게 고차 모델이 선택되지 않도록 아주 작은 절대 허용폭을 함께 둔다.
        y_scale = max(float(np.nanstd(y)), 1.0)
        tolerance = max(min_rmse * 1.05, min_rmse + y_scale * 1e-10)
        eligible = [d for d in sorted(valid_cv) if valid_cv[d] <= tolerance]
        best_degree = min(eligible)
        selection_note = "교차검증 RMSE와 단순성(5% 규칙)을 기준으로 선택"
    else:
        best_degree = 1
        selection_note = "데이터 수가 적어 과적합을 피하기 위해 선형 모델 사용"

    best = model_results[best_degree]
    y_pred = best["pred"]
    residuals = y - y_pred

    # p-value는 선형 기울기의 유의성일 뿐, 다항모델 전체의 p-value가 아니다.
    lin = stats.linregress(data[x_col].values, data[y_col].values)
    linear_p_value = float(lin.pvalue)
    linear_slope = float(lin.slope)
    linear_intercept = float(lin.intercept)

    sorted_data = data.sort_values(x_col)
    sorted_pred = best["model"].predict(sorted_data[[x_col]].values)

    return {
        "kind": "general",
        "data_raw": raw,
        "data": data,
        "outlier_mask": outlier_mask,
        "outlier_count": outlier_count,
        "model_results": model_results,
        "best_degree": best_degree,
        "best": best,
        "selection_note": selection_note,
        "y_pred": y_pred,
        "residuals": residuals,
        "sorted_data": sorted_data,
        "sorted_pred": sorted_pred,
        "r2": best["train_r2"],
        "adj_r2": best["adj_r2"],
        "rmse": best["train_rmse"],
        "linear_p_value": linear_p_value,
        "linear_slope": linear_slope,
        "linear_intercept": linear_intercept,
        "replicate_summary": repeated_summary(data, x_col, y_col),
        "has_replicates": has_replicates(data, x_col),
    }


# =========================================================
# 4. 캘리브레이션 분석
# =========================================================
def analyze_calibration(df, x_col, y_col):
    data = clean_xy(df, x_col, y_col)
    if len(data) < 3 or data[x_col].nunique() < 2:
        return None

    x = data[x_col].values.astype(float)
    y = data[y_col].values.astype(float)
    lin = stats.linregress(x, y)
    pred = lin.slope * x + lin.intercept
    residuals = y - pred

    sorted_data = data.sort_values(x_col)
    sorted_pred = lin.slope * sorted_data[x_col].values + lin.intercept

    return {
        "kind": "calibration",
        "data": data,
        "slope": float(lin.slope),
        "intercept": float(lin.intercept),
        "r2": float(lin.rvalue ** 2),
        "p_value": float(lin.pvalue),
        "stderr": float(lin.stderr) if lin.stderr is not None else np.nan,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "y_pred": pred,
        "residuals": residuals,
        "sorted_data": sorted_data,
        "sorted_pred": sorted_pred,
    }


# =========================================================
# 5. 재료/분광/열분석 특성값 추출
# =========================================================
def analyze_curve(df, x_col, y_col):
    data = clean_xy(df, x_col, y_col).sort_values(x_col)
    if len(data) < 2:
        return None
    return {"kind": "curve", "data": data}


def linear_region_metrics(data, x_col, y_col, x_start, x_end):
    region = data[(data[x_col] >= x_start) & (data[x_col] <= x_end)]
    if len(region) < 3 or region[x_col].nunique() < 2:
        return None
    lin = stats.linregress(region[x_col], region[y_col])
    return {
        "구간 기울기": float(lin.slope),
        "구간 절편": float(lin.intercept),
        "구간 R²": float(lin.rvalue ** 2),
        "구간 데이터 수": int(len(region)),
    }


def material_metrics(result, x_col, y_col, subtype, elastic_region=None):
    data = result["data"]
    x = data[x_col].values.astype(float)
    y = data[y_col].values.astype(float)
    max_idx = int(np.nanargmax(y))
    min_idx = int(np.nanargmin(y))

    metrics = {
        "최대 Y": float(y[max_idx]),
        "최대 Y 발생 X": float(x[max_idx]),
        "최소 Y": float(y[min_idx]),
        "최종 X": float(x[-1]),
        "최종 Y": float(y[-1]),
        "곡선 면적(수치적분)": float(np.trapezoid(y, x)),
    }

    if elastic_region is not None:
        x_start, x_end = elastic_region
        lm = linear_region_metrics(data, x_col, y_col, x_start, x_end)
        if lm:
            metrics.update(lm)

    result["profile"] = subtype
    result["metrics"] = metrics
    return result


def spectrum_metrics(result, x_col, y_col, subtype, signal_mode, prominence_ratio=0.05):
    data = result["data"]
    x = data[x_col].values.astype(float)
    y = data[y_col].values.astype(float)

    y_range = float(np.nanmax(y) - np.nanmin(y))
    prominence = max(y_range * prominence_ratio, np.finfo(float).eps)

    if signal_mode == "아래쪽 피크(예: FT-IR Transmittance)":
        peaks, props = find_peaks(-y, prominence=prominence)
        strengths = props.get("prominences", np.zeros(len(peaks)))
    else:
        peaks, props = find_peaks(y, prominence=prominence)
        strengths = props.get("prominences", np.zeros(len(peaks)))

    peak_table = pd.DataFrame({
        "X": x[peaks] if len(peaks) else [],
        "Y": y[peaks] if len(peaks) else [],
        "prominence": strengths if len(peaks) else [],
    })
    if not peak_table.empty:
        peak_table = peak_table.sort_values("prominence", ascending=False).head(10)

    metrics = {
        "데이터 수": int(len(data)),
        "Y 최대값": float(np.nanmax(y)),
        "Y 최대값 발생 X": float(x[int(np.nanargmax(y))]),
        "Y 최소값": float(np.nanmin(y)),
        "Y 최소값 발생 X": float(x[int(np.nanargmin(y))]),
        "탐지 피크 수": int(len(peaks)),
    }

    result["profile"] = subtype
    result["metrics"] = metrics
    result["peak_table"] = peak_table
    return result


def crossing_x(x, y, target):
    """y가 target을 처음 통과하는 x를 선형보간으로 계산."""
    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - target) == 0:
            return float(x[i])
        if (y1 - target) * (y2 - target) <= 0 and y1 != y2:
            frac = (target - y1) / (y2 - y1)
            return float(x[i] + frac * (x[i + 1] - x[i]))
    return np.nan


def thermal_metrics(result, x_col, y_col, subtype, tga_mass_percent=True, dsc_peak_direction="위쪽 피크"):
    data = result["data"]
    x = data[x_col].values.astype(float)
    y = data[y_col].values.astype(float)

    if subtype == "TGA":
        dydx = np.gradient(y, x)
        dtg_idx = int(np.nanargmin(dydx))
        metrics = {
            "초기 Y": float(y[0]),
            "최종 Y": float(y[-1]),
            "총 변화량(초기-최종)": float(y[0] - y[-1]),
            "최대 감소속도 온도/위치": float(x[dtg_idx]),
            "최대 감소속도 dY/dX": float(dydx[dtg_idx]),
        }
        if tga_mass_percent:
            metrics["T5 (초기값 대비 5%p 감소)"] = crossing_x(x, y, y[0] - 5.0)
            metrics["T10 (초기값 대비 10%p 감소)"] = crossing_x(x, y, y[0] - 10.0)
    else:
        if dsc_peak_direction == "위쪽 피크":
            idx = int(np.nanargmax(y))
        else:
            idx = int(np.nanargmin(y))
        metrics = {
            "주 피크 위치 X": float(x[idx]),
            "주 피크 Y": float(y[idx]),
            "Y 최대값": float(np.nanmax(y)),
            "Y 최소값": float(np.nanmin(y)),
        }

    result["profile"] = subtype
    result["metrics"] = metrics
    return result


# =========================================================
# 6. 세줄요약 생성
# =========================================================
def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "N/A"
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def _p_phrase(p):
    if p is None or not np.isfinite(p):
        return "p=N/A"
    return "p < 0.0001" if p < 0.0001 else f"p={p:.4f}"


def _fitted_trend_text(r, x_label, y_label):
    """선택 모델의 관측 범위 내 예측값을 이용해 단조/비선형 경향을 보수적으로 요약."""
    pred = np.asarray(r["sorted_pred"], dtype=float)
    if len(pred) < 2 or not np.all(np.isfinite(pred)):
        return f"{x_label}와 {y_label}의 전체 추세를 안정적으로 판정하기 어렵습니다."

    diffs = np.diff(pred)
    scale = max(float(np.ptp(pred)), 1.0)
    tol = scale * 1e-6
    sig = diffs[np.abs(diffs) > tol]

    if len(sig) == 0:
        return f"{x_label}가 변해도 {y_label}의 뚜렷한 변화가 거의 나타나지 않습니다."

    pos_ratio = float(np.mean(sig > 0))
    neg_ratio = float(np.mean(sig < 0))

    if r["best_degree"] == 1:
        if r["linear_slope"] > tol:
            return f"{x_label}가 증가할수록 {y_label}가 증가하는 선형 경향이 관찰됩니다."
        if r["linear_slope"] < -tol:
            return f"{x_label}가 증가할수록 {y_label}가 감소하는 선형 경향이 관찰됩니다."
        return f"{x_label}와 {y_label} 사이의 선형 변화는 매우 작습니다."

    if pos_ratio >= 0.90:
        return f"선택된 {r['best_degree']}차 모델에서 {x_label} 증가에 따라 {y_label}가 대체로 증가하는 비선형 경향이 관찰됩니다."
    if neg_ratio >= 0.90:
        return f"선택된 {r['best_degree']}차 모델에서 {x_label} 증가에 따라 {y_label}가 대체로 감소하는 비선형 경향이 관찰됩니다."
    return f"선택된 {r['best_degree']}차 모델에서 증가와 감소가 함께 나타나 단순한 비례·반비례 관계로 요약하기 어렵습니다."


def make_three_line_summary(r, mode, x_label, y_label, subtype=None, signal_mode=None):
    """AI 없이 계산 결과만으로 생성되는 발표/보고서용 세줄요약."""
    if mode == "general":
        line1 = "추세: " + _fitted_trend_text(r, x_label, y_label)

        cv_text = fmt(r["best"]["cv_rmse"])
        if r["best_degree"] == 1:
            line2 = (
                f"신뢰도: R²={fmt(r['r2'])}, Adjusted R²={fmt(r['adj_r2'])}, "
                f"CV RMSE={cv_text}, 선형 기울기 {_p_phrase(r['linear_p_value'])}입니다."
            )
        else:
            line2 = (
                f"신뢰도: R²={fmt(r['r2'])}, Adjusted R²={fmt(r['adj_r2'])}, CV RMSE={cv_text}이며, "
                f"선형 {_p_phrase(r['linear_p_value'])}는 선택된 비선형 모델 전체의 유의성 검정이 아닙니다."
            )

        if r["has_replicates"]:
            groups = len(r["replicate_summary"])
            line3 = (
                f"해석: 반복측정이 {groups}개 X 수준에서 확인되므로 평균±SD와 원자료를 함께 보고, "
                "회귀 관계만으로 인과관계를 단정하지 않는 것이 적절합니다."
            )
        else:
            line3 = (
                "해석: 잔차와 원자료 분포를 함께 확인해야 하며, 높은 R²만으로 물리적 타당성이나 인과관계를 단정할 수 없습니다."
            )
        return [line1, line2, line3]

    if mode == "calibration":
        line1 = (
            f"보정식: {y_label} = {r['slope']:.6g} × {x_label} + {r['intercept']:.6g} 로 계산되었습니다."
        )
        line2 = (
            f"적합도: R²={fmt(r['r2'])}, RMSE={fmt(r['rmse'])}, 선형 기울기 {_p_phrase(r['p_value'])}입니다."
        )
        line3 = (
            "해석: 보정식 사용 전 표준점의 측정 범위, 잔차 패턴과 외삽 여부를 함께 확인해야 합니다."
        )
        return [line1, line2, line3]

    metrics = r.get("metrics", {})

    if mode == "material":
        max_y = metrics.get("최대 Y", np.nan)
        max_x = metrics.get("최대 Y 발생 X", np.nan)
        line1 = f"핵심값: 최대 {y_label}는 {fmt(max_y)}이며 {x_label}={fmt(max_x)}에서 관찰되었습니다."
        if "구간 기울기" in metrics:
            line2 = (
                f"곡선특성: 지정한 초기 구간의 기울기는 {fmt(metrics.get('구간 기울기'))}, "
                f"구간 R²는 {fmt(metrics.get('구간 R²'))}입니다."
            )
        else:
            line2 = (
                f"곡선특성: 수치적분 면적은 {fmt(metrics.get('곡선 면적(수치적분)'))}, "
                f"최종 {x_label}는 {fmt(metrics.get('최종 X'))}입니다."
            )
        line3 = (
            "해석: 자동 계산값은 곡선의 1차 정리값이며, 탄성계수·항복강도·파단특성의 최종 보고는 시험 규격과 단위 정의를 확인해야 합니다."
        )
        return [line1, line2, line3]

    if mode == "spectroscopy":
        peak_df = r.get("peak_table", pd.DataFrame())
        if peak_df is not None and not peak_df.empty:
            top = peak_df.head(3)
            peak_positions = ", ".join(fmt(v, 2) for v in top["X"].tolist())
            strongest = top.iloc[0]
            line1 = f"피크: 현재 민감도에서 {metrics.get('탐지 피크 수', 0)}개가 탐지되었고, 주요 {x_label} 위치는 {peak_positions}입니다."
            line2 = (
                f"가장 두드러진 탐지 피크는 {x_label}={fmt(strongest['X'], 2)}, "
                f"{y_label}={fmt(strongest['Y'], 4)}이며 prominence={fmt(strongest['prominence'], 4)}입니다."
            )
        else:
            line1 = "피크: 현재 설정한 민감도에서는 뚜렷한 피크가 탐지되지 않았습니다."
            line2 = (
                f"범위: {y_label} 최대값은 {fmt(metrics.get('Y 최대값'))}, 최소값은 {fmt(metrics.get('Y 최소값'))}입니다."
            )
        line3 = (
            f"해석: {subtype or '분광'}의 자동 피크 탐지는 위치 후보를 정리하는 기능이며, 작용기·전자전이 등의 assignment는 시료 조성과 문헌을 근거로 별도 확인해야 합니다."
        )
        return [line1, line2, line3]

    if mode == "thermal":
        if subtype == "TGA":
            t5 = metrics.get("T5 (초기값 대비 5%p 감소)", np.nan)
            t10 = metrics.get("T10 (초기값 대비 10%p 감소)", np.nan)
            if np.isfinite(t5) or np.isfinite(t10):
                line1 = f"열안정성: T5={fmt(t5, 2)}, T10={fmt(t10, 2)}로 계산되었습니다."
            else:
                line1 = (
                    f"질량변화: 초기 {y_label}={fmt(metrics.get('초기 Y'))}, 최종 {y_label}={fmt(metrics.get('최종 Y'))}입니다."
                )
            line2 = (
                f"분해특성: 최대 감소속도는 {x_label}={fmt(metrics.get('최대 감소속도 온도/위치'), 2)}에서 나타났고, "
                f"dY/dX={fmt(metrics.get('최대 감소속도 dY/dX'))}입니다."
            )
            line3 = (
                "해석: T5/T10과 최대 감소속도 위치는 자동 계산 보조값이며, onset·잔존량·분해단계의 최종 판단은 baseline과 장비 분석 조건을 확인해야 합니다."
            )
        else:
            line1 = (
                f"주 피크: {x_label}={fmt(metrics.get('주 피크 위치 X'), 2)}에서 {y_label}={fmt(metrics.get('주 피크 Y'))}의 주 피크가 관찰되었습니다."
            )
            line2 = (
                f"신호범위: {y_label} 최대값={fmt(metrics.get('Y 최대값'))}, 최소값={fmt(metrics.get('Y 최소값'))}입니다."
            )
            line3 = (
                "해석: 주 피크 위치는 1차 탐색값이며, Tg·Tm·Tc·ΔH 등의 최종 값은 baseline, 피크 방향과 적분 범위를 확인한 뒤 확정해야 합니다."
            )
        return [line1, line2, line3]

    return [
        "요약: 현재 분석 유형의 핵심 결과를 계산했습니다.",
        "신뢰도: 원자료와 자동 추출값을 함께 확인하세요.",
        "해석: 최종 보고값은 실험 조건과 분석 기준을 확인한 뒤 확정하세요.",
    ]


def render_three_line_summary(lines, heading="세줄요약"):
    st.subheader(heading)
    labels = ["1. 핵심", "2. 근거", "3. 해석"]
    for label, line in zip(labels, lines):
        st.markdown(f"**{label}**  {line}")


# =========================================================
# 7. 차트 생성
# =========================================================
def add_general_traces(fig, r, x_col, y_col, color, show_errorbars=True, show_raw=True):
    if show_raw:
        fig.add_trace(go.Scatter(
            x=r["data"][x_col], y=r["data"][y_col],
            mode="markers", marker=dict(color=color, size=7, opacity=0.55),
            name="원자료",
        ))

    if show_errorbars and r["has_replicates"]:
        s = r["replicate_summary"]
        fig.add_trace(go.Scatter(
            x=s[x_col], y=s["mean"],
            mode="markers+lines",
            error_y=dict(type="data", array=s["sd"].fillna(0), visible=True),
            marker=dict(color=color, size=9),
            line=dict(color=color),
            name="평균 ± SD",
        ))

    fig.add_trace(go.Scatter(
        x=r["sorted_data"][x_col], y=r["sorted_pred"],
        mode="lines", line=dict(color="red", width=2),
        name=f"선택 모델 ({r['best_degree']}차)",
    ))


def add_calibration_traces(fig, r, x_col, y_col, color):
    fig.add_trace(go.Scatter(
        x=r["data"][x_col], y=r["data"][y_col], mode="markers",
        marker=dict(color=color, size=8, opacity=0.7), name="측정값",
    ))
    fig.add_trace(go.Scatter(
        x=r["sorted_data"][x_col], y=r["sorted_pred"], mode="lines",
        line=dict(color="red", width=2), name="선형 보정선",
    ))


def add_curve_trace(fig, r, x_col, y_col, color, name):
    fig.add_trace(go.Scatter(
        x=r["data"][x_col], y=r["data"][y_col],
        mode="lines", line=dict(color=color, width=2), name=name,
    ))


def matplotlib_chart(r, x_col, y_col, x_label, y_label, mode):
    fig, ax = plt.subplots(figsize=(7, 4))

    if mode == "general":
        ax.scatter(r["data"][x_col], r["data"][y_col], alpha=0.55, s=24, label="Raw data")
        if r["has_replicates"]:
            s = r["replicate_summary"]
            ax.errorbar(s[x_col], s["mean"], yerr=s["sd"].fillna(0), fmt="o-", capsize=3, label="Mean +/- SD")
        ax.plot(r["sorted_data"][x_col], r["sorted_pred"], linewidth=1.5, label=f"Selected model ({r['best_degree']})")
    elif mode == "calibration":
        ax.scatter(r["data"][x_col], r["data"][y_col], alpha=0.7, s=28, label="Data")
        ax.plot(r["sorted_data"][x_col], r["sorted_pred"], linewidth=1.5, label="Linear fit")
    else:
        ax.plot(r["data"][x_col], r["data"][y_col], linewidth=1.5, label="Curve")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


def matplotlib_residual_chart(r):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(r["y_pred"], r["residuals"], alpha=0.6, s=25)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Predicted")
    ax.grid(True, alpha=0.25)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


# =========================================================
# 8. PDF 생성
# =========================================================
def make_pdf(
    r, mode, data_type_key, x_col, y_col, x_label, y_label,
    ai_interpretation=None, subtype=None, signal_mode=None
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    title_style = ParagraphStyle(
        "KTitle", fontName=FONT_NAME, fontSize=18,
        spaceAfter=6, textColor=colors.HexColor("#1a1a2e"), leading=24,
    )
    h1_style = ParagraphStyle(
        "KH1", fontName=FONT_NAME, fontSize=13,
        spaceBefore=14, spaceAfter=5,
        textColor=colors.HexColor("#16213e"), leading=18,
    )
    body_style = ParagraphStyle(
        "KBody", fontName=FONT_NAME, fontSize=9.5,
        spaceAfter=4, leading=15,
    )
    caption_style = ParagraphStyle(
        "KCaption", fontName=FONT_NAME, fontSize=9,
        textColor=colors.grey, spaceAfter=8, leading=13,
    )

    story = [
        Paragraph("실험 데이터 분석 리포트", title_style),
        Paragraph(
            f"유형: {data_type_key} | X: {x_label} | Y: {y_label}",
            caption_style,
        ),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#4C72B0"), spaceAfter=10),
    ]

    rows = [["항목", "값"]]
    if mode == "general":
        rows += [
            ["데이터 수", str(len(r["data"]))],
            ["선택 모델", f"{r['best_degree']}차"],
            ["선택 기준", r["selection_note"]],
            ["R²", fmt(r["r2"])],
            ["Adjusted R²", fmt(r["adj_r2"])],
            ["CV RMSE", fmt(r["best"]["cv_rmse"])],
            ["훈련 RMSE", fmt(r["rmse"])],
            ["선형 추세 p-value", fmt(r["linear_p_value"])],
            ["반복측정", "있음" if r["has_replicates"] else "없음"],
        ]
    elif mode == "calibration":
        rows += [
            ["데이터 수", str(len(r["data"]))],
            ["기울기", fmt(r["slope"])],
            ["절편", fmt(r["intercept"])],
            ["R²", fmt(r["r2"])],
            ["p-value", fmt(r["p_value"])],
            ["RMSE", fmt(r["rmse"])],
        ]
    else:
        rows += [["데이터 수", str(len(r["data"]))]]
        for key, value in r.get("metrics", {}).items():
            rows.append([str(key), fmt(value) if isinstance(value, (int, float, np.number)) else str(value)])

    table = Table(rows, colWidths=[6 * cm, 9 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    summary_lines = make_three_line_summary(r, mode, x_label, y_label, subtype, signal_mode)
    story.append(Paragraph("1. 세줄요약", h1_style))
    for idx, line in enumerate(summary_lines, 1):
        safe_line = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(f"{idx}. {safe_line}", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. 분석 요약", h1_style))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("3. 그래프", h1_style))
    story.append(RLImage(matplotlib_chart(r, x_col, y_col, x_label, y_label, mode), width=14 * cm, height=8 * cm))

    if mode in ("general", "calibration"):
        story.append(Paragraph("4. 잔차 플롯", h1_style))
        story.append(RLImage(matplotlib_residual_chart(r), width=14 * cm, height=7 * cm))

    if mode == "general":
        story.append(Paragraph("5. 해석 주의사항", h1_style))
        story.append(Paragraph(
            "표시된 p-value는 선형 기울기에 대한 값입니다. 선택 모델이 2차 또는 3차일 경우 "
            "이 p-value를 비선형 모델 전체의 유의성으로 해석하면 안 됩니다.",
            body_style,
        ))
        if ai_interpretation:
            story.append(Paragraph("6. AI 보조 해석", h1_style))
            for line in ai_interpretation.split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), body_style))

    if mode in ("material", "spectroscopy", "thermal"):
        story.append(Paragraph("4. 해석 주의사항", h1_style))
        story.append(Paragraph(
            "자동 추출값은 원자료 정리와 1차 검토를 위한 보조값입니다. 장비 소프트웨어의 baseline, "
            "onset, integration, smoothing 설정과 분석 표준에 따라 최종 보고값이 달라질 수 있으므로 "
            "논문 또는 공식 보고서에는 원 장비 분석 조건을 함께 확인해야 합니다.",
            body_style,
        ))

    doc.build(story)
    buf.seek(0)
    return buf


# =========================================================
# 9. Streamlit UI
# =========================================================
st.set_page_config(page_title="세줄요약좀 - 실험 데이터", layout="wide")
st.title("세줄요약좀")
st.caption("실험 데이터 시각화 · 반복실험 요약 · 특성값 1차 추출")

with st.sidebar:
    st.header("설정")
    multi_mode = st.toggle("멀티 파일 비교 모드", value=False)
    st.markdown("---")

    if multi_mode:
        files = st.file_uploader(
            "CSV 파일 업로드 (여러 개 선택 가능)",
            type="csv", accept_multiple_files=True,
        )
    else:
        single_file = st.file_uploader("CSV 파일 업로드", type="csv")
        files = [single_file] if single_file else []

if not files:
    st.info("사이드바에서 CSV 파일을 업로드해 주세요.")
    st.stop()

loaded = []
for f in files:
    f.seek(0, 2)
    size_mb = f.tell() / (1024 * 1024)
    f.seek(0)
    if size_mb > MAX_MB:
        st.sidebar.error(f"{f.name}: {size_mb:.1f}MB — {MAX_MB}MB 초과")
        continue
    df_loaded, enc = load_csv(f)
    if df_loaded is None:
        st.sidebar.error(f"{f.name}: CSV 인코딩 또는 형식 인식 실패")
        continue
    if len(df_loaded) > MAX_ROWS:
        st.sidebar.error(f"{f.name}: {len(df_loaded):,}행 — {MAX_ROWS:,}행 초과")
        continue
    loaded.append({"name": f.name, "df": df_loaded, "enc": enc})
    st.sidebar.success(f"{f.name} ({enc})")

if not loaded:
    st.error("유효한 파일이 없습니다.")
    st.stop()

if multi_mode and len(loaded) > 1:
    common_num_cols = set(loaded[0]["df"].select_dtypes(include=np.number).columns)
    for item in loaded[1:]:
        common_num_cols &= set(item["df"].select_dtypes(include=np.number).columns)
    numeric_cols = sorted(common_num_cols)
else:
    numeric_cols = loaded[0]["df"].select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("공통 숫자형 컬럼이 최소 2개 이상 필요합니다.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("데이터 유형")
    data_type_key = st.selectbox("실험 유형 선택", list(DATA_TYPE_OPTIONS.keys()))
    mode = DATA_TYPE_OPTIONS[data_type_key]["mode"]
    st.caption(DATA_TYPE_OPTIONS[data_type_key]["desc"])

    st.markdown("---")
    st.subheader("축 설정")
    x_col = st.selectbox("X축 선택", numeric_cols)
    y_candidates = [c for c in numeric_cols if c != x_col]
    y_col = st.selectbox("Y축 선택", y_candidates)
    x_unit = st.text_input("X축 단위 (선택)", placeholder="예: °C, nm, cm⁻¹, strain")
    y_unit = st.text_input("Y축 단위 (선택)", placeholder="예: MPa, %, Absorbance")
    x_label = axis_label(x_col, x_unit)
    y_label = axis_label(y_col, y_unit)

    # 모드별 옵션
    remove_outliers = False
    zscore_threshold = 3.0
    show_errorbars = True
    subtype = None
    elastic_region = None
    signal_mode = None
    prominence_ratio = 0.05
    tga_mass_percent = True
    dsc_peak_direction = "위쪽 피크"

    if mode == "general":
        st.markdown("---")
        st.subheader("일반 통계 옵션")
        show_errorbars = st.checkbox("반복측정이면 평균 ± SD 표시", value=True)
        remove_outliers = st.checkbox("Z-score 이상치 제거", value=False)
        zscore_threshold = st.slider("Z-score 임계값", 2.0, 4.0, 3.0, 0.1)
        st.caption("이상치 제거는 일반 실험 모드에만 적용됩니다. 원자료 검토 없이 자동 삭제하지 않는 것을 권장합니다.")

    elif mode == "material":
        st.markdown("---")
        st.subheader("재료 시험 설정")
        subtype = st.selectbox("시험 종류", ["인장", "압축", "굽힘", "기타 곡선"])
        use_region = st.checkbox("초기 선형구간 기울기 계산", value=False)
        if use_region:
            base = clean_xy(loaded[0]["df"], x_col, y_col)
            if not base.empty:
                x_min = float(base[x_col].min())
                x_max = float(base[x_col].max())
                default_end = x_min + (x_max - x_min) * 0.15
                region_start = st.number_input("선형구간 X 시작", value=x_min, format="%.6f")
                region_end = st.number_input("선형구간 X 끝", value=float(default_end), format="%.6f")
                if region_end > region_start:
                    elastic_region = (region_start, region_end)

    elif mode == "spectroscopy":
        st.markdown("---")
        st.subheader("분광 분석 설정")
        subtype = st.selectbox("분석 종류", ["FT-IR", "UV-Vis"])
        signal_mode = st.selectbox(
            "피크 방향",
            ["위쪽 피크(예: Absorbance)", "아래쪽 피크(예: FT-IR Transmittance)"],
        )
        prominence_ratio = st.slider("피크 탐지 민감도", 0.01, 0.30, 0.05, 0.01)
        st.caption("값이 작을수록 작은 피크도 더 많이 탐지합니다. 최종 peak assignment는 직접 확인해야 합니다.")

    elif mode == "thermal":
        st.markdown("---")
        st.subheader("열분석 설정")
        subtype = st.selectbox("분석 종류", ["TGA", "DSC"])
        if subtype == "TGA":
            tga_mass_percent = st.checkbox("Y축이 질량 백분율(%)", value=True)
        else:
            dsc_peak_direction = st.selectbox("주 피크 방향", ["위쪽 피크", "아래쪽 피크"])

    st.markdown("---")
    run_ai = False
    if mode == "general" and not multi_mode:
        st.subheader("AI 보조 해석")
        run_ai = st.button("AI 자동 해석 생성", use_container_width=True)

# 분석 실행
results = []
with st.spinner("분석 중..."):
    for item in loaded:
        if mode == "general":
            r = analyze_general(item["df"], x_col, y_col, remove_outliers, zscore_threshold)
        elif mode == "calibration":
            r = analyze_calibration(item["df"], x_col, y_col)
        else:
            r = analyze_curve(item["df"], x_col, y_col)
            if r is not None and mode == "material":
                r = material_metrics(r, x_col, y_col, subtype, elastic_region)
            elif r is not None and mode == "spectroscopy":
                r = spectrum_metrics(r, x_col, y_col, subtype, signal_mode, prominence_ratio)
            elif r is not None and mode == "thermal":
                r = thermal_metrics(r, x_col, y_col, subtype, tga_mass_percent, dsc_peak_direction)

        if r is None:
            st.warning(f"{item['name']}: 분석 가능한 데이터가 부족합니다.")
            continue
        r["name"] = item["name"]
        r["df"] = item["df"]
        results.append(r)

if not results:
    st.error("분석 가능한 파일이 없습니다.")
    st.stop()

# AI 해석 결과가 다른 데이터에 섞이지 않도록 분석 서명 관리
extra_settings = (
    mode, remove_outliers, zscore_threshold, show_errorbars, subtype,
    elastic_region, signal_mode, prominence_ratio, tga_mass_percent,
    dsc_peak_direction, x_unit, y_unit,
)
current_signature = file_signature(loaded, x_col, y_col, data_type_key, extra_settings)
if st.session_state.get("analysis_signature") != current_signature:
    st.session_state.analysis_signature = current_signature
    st.session_state.ai_interpretation = None

if "ai_interpretation" not in st.session_state:
    st.session_state.ai_interpretation = None

if run_ai and mode == "general" and not multi_mode:
    r = results[0]
    p = r["linear_p_value"]
    p_text = "< 0.0001" if p < 0.0001 else f"{p:.4f}"
    best_cv = r["best"]["cv_rmse"]
    model_warning = (
        "선택 모델은 선형이며, 아래 p-value는 선택 모델의 선형 기울기 검정과 일치합니다."
        if r["best_degree"] == 1
        else "선택 모델은 비선형입니다. 아래 p-value는 비선형 모델 전체가 아니라 '선형 추세의 기울기'에 대한 값입니다."
    )

    prompt = f"""당신은 실험 데이터 분석 보조자입니다. 아래 수치를 과장 없이 한국어로 해석하세요.

[분석 정보]
- X: {x_label}
- Y: {y_label}
- 데이터 수: {len(r['data'])}
- 반복측정 존재: {r['has_replicates']}
- 선택 모델: {r['best_degree']}차
- 모델 선택 방식: {r['selection_note']}
- R²: {r['r2']:.4f}
- Adjusted R²: {fmt(r['adj_r2'])}
- CV RMSE: {fmt(best_cv)}
- 훈련 RMSE: {r['rmse']:.4f}
- 선형 추세 p-value: {p_text}
- 주의: {model_warning}

반드시 다음 원칙을 지키세요.
1. 상관관계를 인과관계로 표현하지 마세요.
2. 2차/3차 모델일 때 선형 p-value를 모델 전체의 유의성으로 표현하지 마세요.
3. R²가 높다는 이유만으로 물리적 타당성을 단정하지 마세요.
4. 결과 요약, 한계, 추가 확인사항 순으로 짧게 작성하세요.
"""

    try:
        import requests as req
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.session_state.ai_interpretation = "API 키가 설정되지 않았습니다."
        else:
            with st.spinner("AI 해석 중..."):
                resp = req.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 900,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                st.session_state.ai_interpretation = resp.json()["content"][0]["text"]
    except Exception as e:
        st.session_state.ai_interpretation = f"AI 해석 오류: {e}"


# =========================================================
# 10. 멀티 파일 모드
# =========================================================
if multi_mode:
    st.warning(
        "멀티 파일 비교는 모든 파일의 X/Y가 같은 물리량·단위·측정조건을 의미할 때만 직접 비교하세요."
    )

    tab_graph, tab_three, tab_summary, tab_download = st.tabs(["그래프 비교", "세줄요약", "요약 비교", "다운로드"])

    with tab_graph:
        st.subheader(f"{x_label} vs {y_label}")
        overlay = st.checkbox("한 그래프에 겹쳐 보기", value=(mode != "general"))

        if overlay:
            fig = go.Figure()
            for i, r in enumerate(results):
                color = COLORS[i % len(COLORS)]
                if mode == "general":
                    fig.add_trace(go.Scatter(
                        x=r["data"][x_col], y=r["data"][y_col],
                        mode="markers", marker=dict(color=color, size=6, opacity=0.55),
                        name=r["name"],
                    ))
                elif mode == "calibration":
                    fig.add_trace(go.Scatter(
                        x=r["data"][x_col], y=r["data"][y_col], mode="markers",
                        marker=dict(color=color, size=7), name=f"{r['name']} data",
                    ))
                    fig.add_trace(go.Scatter(
                        x=r["sorted_data"][x_col], y=r["sorted_pred"], mode="lines",
                        line=dict(color=color, width=2), name=f"{r['name']} fit",
                    ))
                else:
                    add_curve_trace(fig, r, x_col, y_col, color, r["name"])
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            for row_start in range(0, len(results), 2):
                row = results[row_start:row_start + 2]
                cols = st.columns(len(row))
                for j, (col, r) in enumerate(zip(cols, row)):
                    with col:
                        color = COLORS[(row_start + j) % len(COLORS)]
                        fig = go.Figure()
                        if mode == "general":
                            add_general_traces(fig, r, x_col, y_col, color, show_errorbars)
                        elif mode == "calibration":
                            add_calibration_traces(fig, r, x_col, y_col, color)
                        else:
                            add_curve_trace(fig, r, x_col, y_col, color, r["name"])
                        fig.update_layout(
                            title=r["name"], xaxis_title=x_label, yaxis_title=y_label,
                            height=360, margin=dict(t=45, b=40, l=45, r=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab_three:
        st.subheader("파일별 세줄요약")
        for idx, r in enumerate(results):
            if idx > 0:
                st.markdown("---")
            st.markdown(f"### {r['name']}")
            lines = make_three_line_summary(r, mode, x_label, y_label, subtype, signal_mode)
            render_three_line_summary(lines, heading="세줄요약")

    with tab_summary:
        st.subheader("파일별 핵심 지표")
        rows = []
        for r in results:
            if mode == "general":
                rows.append({
                    "파일": r["name"],
                    "선택 모델": f"{r['best_degree']}차",
                    "R²": fmt(r["r2"]),
                    "Adjusted R²": fmt(r["adj_r2"]),
                    "CV RMSE": fmt(r["best"]["cv_rmse"]),
                    "선형 추세 p-value": fmt(r["linear_p_value"]),
                    "반복측정": "있음" if r["has_replicates"] else "없음",
                })
            elif mode == "calibration":
                rows.append({
                    "파일": r["name"], "기울기": fmt(r["slope"]),
                    "절편": fmt(r["intercept"]), "R²": fmt(r["r2"]),
                    "p-value": fmt(r["p_value"]), "RMSE": fmt(r["rmse"]),
                })
            else:
                row = {"파일": r["name"]}
                for k, v in r.get("metrics", {}).items():
                    row[k] = fmt(v) if isinstance(v, (int, float, np.number)) else v
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_download:
        st.subheader("결과 다운로드")
        for r in results:
            if mode == "general":
                out_df = r["data"].copy()
                out_df["모델 예측값"] = r["y_pred"]
                out_df["잔차"] = r["residuals"]
            elif mode == "calibration":
                out_df = r["data"].copy()
                out_df["선형 예측값"] = r["y_pred"]
                out_df["잔차"] = r["residuals"]
            else:
                out_df = r["data"].copy()

            name = r["name"].rsplit(".", 1)[0]
            st.download_button(
                f"📥 {r['name']} 결과 CSV",
                data=out_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{name}_result.csv", mime="text/csv",
                use_container_width=True,
            )
            file_summary_lines = make_three_line_summary(r, mode, x_label, y_label, subtype, signal_mode)
            file_summary_text = "\n".join(f"{i}. {line}" for i, line in enumerate(file_summary_lines, 1))
            st.download_button(
                f"📥 {r['name']} 세줄요약 TXT",
                data=file_summary_text.encode("utf-8-sig"),
                file_name=f"{name}_three_line_summary.txt", mime="text/plain",
                use_container_width=True,
            )


# =========================================================
# 11. 싱글 파일 모드
# =========================================================
else:
    r = results[0]

    if mode == "general":
        tabs = st.tabs(["데이터", "분석 결과", "세줄요약", "AI 해석", "다운로드"])
    else:
        tabs = st.tabs(["데이터", "그래프 / 특성값", "세줄요약", "다운로드"])

    with tabs[0]:
        st.subheader("업로드 데이터")
        st.caption(f"{r['name']} | {len(r['df']):,}행 × {len(r['df'].columns)}열")
        st.dataframe(r["df"], use_container_width=True)

        if mode == "general" and r["outlier_count"] > 0:
            status = "제거됨" if remove_outliers else "탐지만 됨"
            st.warning(f"Z-score 기준 이상치 후보 {r['outlier_count']}개 — 현재 {status}")

        if mode == "general" and r["has_replicates"]:
            st.markdown("---")
            st.subheader("반복측정 요약")
            s = r["replicate_summary"].copy()
            st.dataframe(s, use_container_width=True, hide_index=True)

    with tabs[1]:
        fig = go.Figure()
        if mode == "general":
            add_general_traces(fig, r, x_col, y_col, COLORS[0], show_errorbars)
        elif mode == "calibration":
            add_calibration_traces(fig, r, x_col, y_col, COLORS[0])
        else:
            add_curve_trace(fig, r, x_col, y_col, COLORS[0], r["name"])

        fig.update_layout(
            title=f"{x_label} vs {y_label}",
            xaxis_title=x_label, yaxis_title=y_label,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        if mode == "general":
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("선택 모델", f"{r['best_degree']}차")
            c2.metric("R²", fmt(r["r2"]))
            c3.metric("Adjusted R²", fmt(r["adj_r2"]))
            c4.metric("CV RMSE", fmt(r["best"]["cv_rmse"]))
            st.caption(r["selection_note"])

            st.markdown("---")
            st.subheader("모델 비교")
            model_rows = []
            for degree, mr in r["model_results"].items():
                model_rows.append({
                    "모델": f"{degree}차",
                    "훈련 R²": fmt(mr["train_r2"]),
                    "Adjusted R²": fmt(mr["adj_r2"]),
                    "훈련 RMSE": fmt(mr["train_rmse"]),
                    "CV RMSE": fmt(mr["cv_rmse"]),
                    "선택": "✅" if degree == r["best_degree"] else "",
                })
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            p = r["linear_p_value"]
            p_text = "< 0.0001" if p < 0.0001 else f"{p:.4f}"
            st.subheader("선형 추세 검정")
            cp1, cp2 = st.columns(2)
            cp1.metric("선형 기울기", fmt(r["linear_slope"]))
            cp2.metric("선형 추세 p-value", p_text)
            if r["best_degree"] == 1:
                st.caption("선택 모델이 선형이므로 이 p-value는 선형 기울기 검정과 직접 대응합니다.")
            else:
                st.warning(
                    "선택 모델이 비선형입니다. 위 p-value는 2차/3차 모델 전체의 유의성이 아니라 "
                    "'선형 추세의 기울기'에 대한 값입니다."
                )

            st.markdown("---")
            st.subheader("잔차 플롯")
            residual_fig = go.Figure()
            residual_fig.add_trace(go.Scatter(
                x=r["y_pred"], y=r["residuals"], mode="markers",
                marker=dict(size=7, opacity=0.65), name="잔차",
            ))
            residual_fig.add_hline(y=0, line_dash="dash")
            residual_fig.update_layout(xaxis_title="예측값", yaxis_title="잔차", height=350)
            st.plotly_chart(residual_fig, use_container_width=True)

        elif mode == "calibration":
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("기울기", fmt(r["slope"]))
            c2.metric("절편", fmt(r["intercept"]))
            c3.metric("R²", fmt(r["r2"]))
            c4.metric("RMSE", fmt(r["rmse"]))
            st.code(f"{y_col} = {r['slope']:.6g} × {x_col} + {r['intercept']:.6g}")
            st.caption("캘리브레이션 모드는 선형 보정을 전제로 한 경우에 사용하세요.")

        else:
            st.markdown("---")
            st.subheader("자동 추출 특성값")
            metric_rows = []
            for key, value in r.get("metrics", {}).items():
                metric_rows.append({"항목": key, "값": fmt(value) if isinstance(value, (int, float, np.number)) else value})
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

            if mode == "spectroscopy" and "peak_table" in r:
                st.markdown("---")
                st.subheader("탐지 피크 (최대 10개)")
                peak_df = r["peak_table"].copy()
                if peak_df.empty:
                    st.info("현재 민감도에서 탐지된 피크가 없습니다.")
                else:
                    peak_df = peak_df.rename(columns={"X": x_label, "Y": y_label})
                    st.dataframe(peak_df, use_container_width=True, hide_index=True)

            st.info(
                "특성값 자동 추출은 1차 정리용입니다. FT-IR peak assignment, TGA onset, DSC enthalpy, "
                "탄성계수 등 최종 보고값은 장비 조건과 분석 기준을 확인한 뒤 확정하세요."
            )

    summary_lines = make_three_line_summary(r, mode, x_label, y_label, subtype, signal_mode)
    with tabs[2]:
        render_three_line_summary(summary_lines)
        st.caption("세줄요약은 AI가 아니라 현재 계산된 분석 결과만으로 자동 생성됩니다.")

    if mode == "general":
        with tabs[3]:
            st.subheader("AI 보조 해석")
            if st.session_state.ai_interpretation:
                st.markdown(st.session_state.ai_interpretation)
            else:
                st.info("사이드바의 'AI 자동 해석 생성' 버튼을 누르면 현재 분석에 대한 보조 해석이 생성됩니다.")

        download_tab = tabs[4]
    else:
        download_tab = tabs[3]

    with download_tab:
        st.subheader("다운로드")
        ca, cb = st.columns(2)
        with ca:
            if mode == "general":
                result_df = r["data"].copy()
                result_df["모델 예측값"] = r["y_pred"]
                result_df["잔차"] = r["residuals"]
            elif mode == "calibration":
                result_df = r["data"].copy()
                result_df["선형 예측값"] = r["y_pred"]
                result_df["잔차"] = r["residuals"]
            else:
                result_df = r["data"].copy()

            st.download_button(
                "📥 결과 CSV 다운로드",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="experiment_result.csv",
                mime="text/csv", use_container_width=True,
            )

            if mode == "general" and r["has_replicates"]:
                st.download_button(
                    "📥 반복측정 요약 CSV 다운로드",
                    data=r["replicate_summary"].to_csv(index=False).encode("utf-8-sig"),
                    file_name="replicate_summary.csv",
                    mime="text/csv", use_container_width=True,
                )

            summary_text = "\n".join(f"{i}. {line}" for i, line in enumerate(summary_lines, 1))
            st.download_button(
                "📥 세줄요약 TXT 다운로드",
                data=summary_text.encode("utf-8-sig"),
                file_name="three_line_summary.txt",
                mime="text/plain", use_container_width=True,
            )

        with cb:
            if st.button("PDF 리포트 생성", use_container_width=True):
                pdf_buf = make_pdf(
                    r, mode, data_type_key, x_col, y_col, x_label, y_label,
                    ai_interpretation=st.session_state.ai_interpretation if mode == "general" else None,
                    subtype=subtype, signal_mode=signal_mode,
                )
                st.download_button(
                    "📥 PDF 다운로드", data=pdf_buf,
                    file_name="experiment_report.pdf",
                    mime="application/pdf", use_container_width=True,
                )
