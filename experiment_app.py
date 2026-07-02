import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats
import numpy as np
import traceback
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -------------------------
# 데이터 유형 정의
# -------------------------
DATA_TYPE_OPTIONS = {
    "일반 실험": {
        "desc": "변수 간 관계 분석이 목적인 일반 실험 데이터",
        "is_instrument": False,
    },
    "재료 시험 (인장·압축·굽힘 등)": {
        "desc": "하중-변위, 응력-변형률 등 재료 물성 측정 데이터",
        "is_instrument": True,
    },
    "캘리브레이션 / 보정": {
        "desc": "캘리퍼스, 저울, 온도계 등 측정 기구의 보정 데이터",
        "is_instrument": True,
    },
}

# -------------------------
# 한글 폰트 등록
# -------------------------
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

# -------------------------
# 페이지 설정
# -------------------------
st.set_page_config(page_title="데이터 분석", layout="wide")
st.title("데이터 분석")

MAX_ROWS = 50_000
MAX_MB = 10

# -------------------------
# 사이드바
# -------------------------
with st.sidebar:
    st.header("설정")
    file = st.file_uploader("CSV 파일 업로드", type="csv")

if file is None:
    st.info("사이드바에서 CSV 파일을 업로드해 주세요.")
    st.stop()

try:
    # 파일 크기 체크
    file.seek(0, 2)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if file_size_mb > MAX_MB:
        st.error(f"파일이 너무 큽니다. ({MAX_MB}MB 이하만 허용, 현재 {file_size_mb:.1f}MB)")
        st.stop()

    # 인코딩 자동 감지
    encodings = ["utf-8", "cp949", "euc-kr", "utf-8-sig"]
    df = None
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            st.sidebar.success(f"인코딩 감지: {enc}")
            break
        except Exception:
            continue

    if df is None:
        st.error("CSV 인코딩을 인식할 수 없습니다.")
        st.stop()

    if len(df) > MAX_ROWS:
        st.error(f"데이터가 너무 큽니다. ({MAX_ROWS:,}행 이하만 허용)")
        st.stop()

    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("숫자형 컬럼이 최소 2개 이상 필요합니다.")
        st.stop()

    with st.sidebar:
        st.markdown("---")

        # -------------------------
        # 데이터 유형 선택
        # -------------------------
        st.subheader("데이터 유형")
        data_type_key = st.selectbox(
            "실험 유형 선택",
            list(DATA_TYPE_OPTIONS.keys()),
            help="유형에 따라 분석 항목이 달라집니다."
        )
        data_type_info = DATA_TYPE_OPTIONS[data_type_key]
        is_instrument = data_type_info["is_instrument"]
        st.caption(data_type_info["desc"])

        st.markdown("---")
        st.subheader("축 설정")
        x_col = st.selectbox("X축 선택", numeric_cols)
        y_candidates = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Y축 선택", y_candidates)

        st.markdown("---")
        st.subheader("이상치 설정")
        remove_outliers = st.checkbox("이상치 제거", value=False)
        zscore_threshold = st.slider(
            "Z-score 임계값", 2.0, 4.0, 3.0, 0.1,
            help="값이 낮을수록 더 많은 이상치를 제거합니다."
        )

        if not is_instrument:
            st.markdown("---")
            st.subheader("AI 해석")
            run_ai = st.button("AI 자동 해석 생성", use_container_width=True)
        else:
            run_ai = False

    # -------------------------
    # 데이터 준비 & 이상치 처리
    # -------------------------
    data_raw = df[[x_col, y_col]].dropna()
    if len(data_raw) < 2:
        st.error("유효한 데이터가 2개 이상 필요합니다.")
        st.stop()

    z_scores = np.abs(stats.zscore(data_raw[[x_col, y_col]]))
    outlier_mask = (z_scores > zscore_threshold).any(axis=1)
    outlier_count = outlier_mask.sum()
    data = data_raw[~outlier_mask].copy() if remove_outliers else data_raw.copy()

    X = data[[x_col]].values
    y = data[y_col].values
    sorted_data = data.sort_values(x_col)
    X_sorted = sorted_data[[x_col]].values

    degree_label = {1: "선형 (1차)", 2: "다항식 (2차)", 3: "다항식 (3차)"}

    # -------------------------
    # 회귀 분석 (공통)
    # -------------------------
    with st.spinner("분석 중..."):
        model_results = {}
        for degree in [1, 2, 3]:
            pipe = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            pipe.fit(X, y)
            y_pred_d = pipe.predict(X)
            r2_d = r2_score(y, y_pred_d)

            if not is_instrument and len(X) >= 5:
                cv_k = min(5, len(X))
                cv_scores = cross_val_score(pipe, X, y, cv=cv_k, scoring="r2")
                cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            else:
                cv_mean, cv_std = np.nan, np.nan

            model_results[degree] = {
                "pipe": pipe,
                "r2": r2_d,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "y_pred": y_pred_d,
            }

        lin_slope, lin_intercept, r_val, p_value, std_err = stats.linregress(
            data[x_col].values, data[y_col].values
        )

        def best_score(d):
            cv = model_results[d]["cv_mean"]
            return cv if not np.isnan(cv) else model_results[d]["r2"]

        best_degree = max([1, 2, 3], key=best_score)
        best = model_results[best_degree]
        y_pred = best["y_pred"]
        r2 = best["r2"]
        residuals = y - y_pred

        lin_model = model_results[1]["pipe"]
        slope = float(lin_model.named_steps["linearregression"].coef_[1])
        intercept = float(lin_model.named_steps["linearregression"].intercept_)
        y_line = best["pipe"].predict(X_sorted)

    regression_formula = f"{y_col} = {slope:.4f} x {x_col} + {intercept:.4f}"

    # -------------------------
    # 일반 실험 전용: 통계 지표
    # -------------------------
    if not is_instrument:
        eps = 1e-6
        rmse = np.sqrt(np.mean(residuals ** 2))
        p_text = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
        sig_text = "통계적으로 유의함 (p < 0.05)" if p_value < 0.05 else "통계적으로 유의하지 않음 (p ≥ 0.05)"

        trend_text = (
            "X가 증가할수록 Y도 증가하는 경향" if lin_slope > eps
            else "X가 증가할수록 Y는 감소하는 경향" if lin_slope < -eps
            else "뚜렷한 변화가 관찰되지 않음"
        )
        if r2 < 0:
            validity_text = "평균값 예측보다 성능이 낮습니다."
        elif r2 < 0.7:
            validity_text = "선형 관계가 약합니다."
        elif r2 < 0.9:
            validity_text = "상당한 선형 관계가 존재합니다."
        else:
            validity_text = "매우 강한 선형 관계가 존재합니다."

        if r2 >= 0.9 and lin_slope > eps:
            insight_text = "강한 비례 관계를 가지며 예측 모델로 활용 가능합니다."
        elif r2 >= 0.9 and lin_slope < -eps:
            insight_text = "강한 반비례 관계를 가지며 감소 모델로 해석 가능합니다."
        elif r2 < 0.7:
            insight_text = "비선형 관계 또는 외부 요인의 영향을 고려해야 합니다."
        else:
            insight_text = "일정 수준의 경향성을 가지며 추가 분석이 권장됩니다."

    # -------------------------
    # AI 해석 (일반 실험 전용)
    # -------------------------
    if "ai_interpretation" not in st.session_state:
        st.session_state.ai_interpretation = None

    if run_ai and not is_instrument:
        cv_info = (
            f"{best['cv_mean']:.4f} ± {best['cv_std']:.4f}"
            if not np.isnan(best["cv_mean"]) else "데이터 부족으로 계산 불가"
        )
        prompt = f"""당신은 실험 데이터 분석 전문가입니다. 다음 회귀 분석 결과를 바탕으로 한국어로 학술적이고 명확하게 해석해 주세요.

[분석 정보]
- X 변수: {x_col}
- Y 변수: {y_col}
- 데이터 수: {len(data)}개 (이상치 제거: {outlier_count}개)
- 최적 모델: {degree_label[best_degree]}
- R² (결정계수): {r2:.4f}
- 교차검증 R²: {cv_info}
- RMSE: {rmse:.4f}
- p-value: {p_text} ({sig_text})
- 잔차 평균: {np.mean(residuals):.4f}
- 잔차 표준편차: {np.std(residuals):.4f}

다음 항목을 순서대로 작성해 주세요:
1. 회귀 분석 요약 (최적 모델 포함, 2-3문장)
2. 변수 간 관계 해석 (R², p-value 근거 포함)
3. 모델 적합성 평가 (교차검증 결과, RMSE, 잔차 분포 언급)
4. 한계점 및 추가 분석 제안

각 항목은 번호와 함께 명확히 구분해 주세요."""

        with st.spinner("AI가 분석 결과를 해석하고 있습니다..."):
            try:
                import requests
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    st.session_state.ai_interpretation = (
                        "API 키가 설정되지 않았습니다. "
                        "Streamlit Cloud라면 Settings > Secrets에 ANTHROPIC_API_KEY를 추가해 주세요."
                    )
                else:
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 1000,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.session_state.ai_interpretation = result["content"][0]["text"]
            except requests.HTTPError as e:
                status = e.response.status_code if e.response else "unknown"
                msg = e.response.json().get("error", {}).get("message", str(e)) if e.response else str(e)
                st.session_state.ai_interpretation = f"API 오류 (HTTP {status}): {msg}"
            except Exception as e:
                st.session_state.ai_interpretation = f"AI 해석 생성 중 오류 발생: {e}"

    # -------------------------
    # matplotlib 차트 (PDF용)
    # -------------------------
    def make_scatter_chart():
        fig, ax = plt.subplots(figsize=(7, 4))
        if remove_outliers and outlier_count > 0:
            ax.scatter(
                data_raw[outlier_mask][x_col], data_raw[outlier_mask][y_col],
                alpha=0.4, color="gray", s=25, marker="x", label=f"이상치 ({outlier_count}개)"
            )
        ax.scatter(data[x_col], data[y_col], alpha=0.6, color="#4C72B0", s=30, label="데이터")
        ax.plot(sorted_data[x_col].values, y_line, color="red", linewidth=1.5,
                label=f"회귀선 ({degree_label[best_degree]})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    def make_residual_chart():
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.scatter(y_pred, residuals, alpha=0.6, color="#DD8452", s=30)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("예측값")
        ax.set_ylabel("잔차")
        ax.set_title("잔차 vs 예측값")
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    def make_model_compare_chart():
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = [degree_label[d] for d in [1, 2, 3]]
        r2_vals = [model_results[d]["r2"] for d in [1, 2, 3]]
        bar_colors = ["#e74c3c" if d == best_degree else "#4C72B0" for d in [1, 2, 3]]
        bars = ax.bar(labels, r2_vals, color=bar_colors, alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("R²")
        ax.set_title("모델별 R² 비교 (빨강: 최적 모델)")
        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    # -------------------------
    # PDF 생성
    # -------------------------
    def generate_pdf():
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        title_style = ParagraphStyle(
            "KTitle", fontName=FONT_NAME, fontSize=18,
            spaceAfter=6, textColor=colors.HexColor("#1a1a2e"), leading=24
        )
        h1_style = ParagraphStyle(
            "KH1", fontName=FONT_NAME, fontSize=13,
            spaceBefore=14, spaceAfter=4,
            textColor=colors.HexColor("#16213e"), leading=18
        )
        body_style = ParagraphStyle(
            "KBody", fontName=FONT_NAME, fontSize=10, spaceAfter=4, leading=16
        )
        caption_style = ParagraphStyle(
            "KCaption", fontName=FONT_NAME, fontSize=9,
            textColor=colors.grey, spaceAfter=8, leading=13
        )
        code_style = ParagraphStyle(
            "KCode", fontName="Courier", fontSize=10,
            backColor=colors.HexColor("#f4f4f4"),
            borderPadding=(4, 6, 4, 6), spaceAfter=8, leading=15
        )

        story = []
        story.append(Paragraph("실험 데이터 분석 리포트", title_style))
        story.append(Paragraph(
            f"유형: {data_type_key}  |  변수: {x_col} - {y_col}  |  데이터 수: {len(data)}개",
            caption_style
        ))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=colors.HexColor("#4C72B0"), spaceAfter=12))

        # 공통: 기본 요약 테이블
        story.append(Paragraph("1. 분석 요약", h1_style))
        if is_instrument:
            table_data = [
                ["항목", "값"],
                ["실험 유형", data_type_key],
                ["X 변수", x_col],
                ["Y 변수", y_col],
                ["데이터 수", f"{len(data)}개 (이상치 {outlier_count}개 탐지)"],
                ["회귀선 모델", degree_label[best_degree]],
            ]
        else:
            cv_display = (
                f"{best['cv_mean']:.4f} +/- {best['cv_std']:.4f}"
                if not np.isnan(best["cv_mean"]) else "N/A"
            )
            table_data = [
                ["항목", "값"],
                ["실험 유형", data_type_key],
                ["X 변수", x_col],
                ["Y 변수", y_col],
                ["데이터 수", f"{len(data)}개 (이상치 {outlier_count}개 탐지)"],
                ["최적 모델", degree_label[best_degree]],
                ["결정계수 (R2)", f"{r2:.4f}"],
                ["교차검증 R2", cv_display],
                ["RMSE", f"{rmse:.4f}"],
                ["p-value", f"{p_text}  ({sig_text})"],
            ]

        t = Table(table_data, colWidths=[5*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        story.append(Paragraph("회귀식 (선형 기준)", h1_style))
        story.append(Paragraph(regression_formula, code_style))

        # 일반 실험 전용: 모델 비교 차트
        if not is_instrument:
            story.append(Paragraph("2. 모델 비교", h1_style))
            story.append(RLImage(make_model_compare_chart(), width=10*cm, height=6*cm))
            story.append(Spacer(1, 6))

        # 공통: 산점도
        sec_num = 3 if not is_instrument else 2
        story.append(Paragraph(f"{sec_num}. 산점도 및 회귀선", h1_style))
        story.append(RLImage(make_scatter_chart(), width=14*cm, height=8*cm))
        story.append(Spacer(1, 6))

        # 일반 실험 전용: 잔차 플롯 + 해석
        if not is_instrument:
            story.append(Paragraph("4. 잔차 플롯", h1_style))
            story.append(RLImage(make_residual_chart(), width=14*cm, height=7*cm))
            story.append(Paragraph(
                "잔차가 0 근처에 고르게 분포할수록 모델 가정이 잘 성립합니다.", caption_style
            ))
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.8,
                                    color=colors.HexColor("#cccccc"), spaceAfter=8))
            story.append(Paragraph("5. 자동 해석", h1_style))
            story.append(Paragraph(f"추세: {trend_text}", body_style))
            story.append(Paragraph(f"타당성: {validity_text}", body_style))
            story.append(Paragraph(f"통계적 유의성: {sig_text}", body_style))
            story.append(Paragraph(f"활용 방향: {insight_text}", body_style))
            if st.session_state.ai_interpretation:
                story.append(Spacer(1, 8))
                story.append(HRFlowable(width="100%", thickness=0.8,
                                        color=colors.HexColor("#cccccc"), spaceAfter=8))
                story.append(Paragraph("6. AI 심층 해석", h1_style))
                for line in st.session_state.ai_interpretation.split("\n"):
                    if line.strip():
                        story.append(Paragraph(line.strip(), body_style))

        doc.build(story)
        buf.seek(0)
        return buf

    # -------------------------
    # 탭 구성 — 유형별 분기
    # -------------------------
    if is_instrument:
        tab_labels = ["데이터", "그래프", "다운로드"]
        tab1, tab2, tab3 = st.tabs(tab_labels)
    else:
        tab_labels = ["데이터", "분석 결과", "AI 해석", "다운로드"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

    # ── 탭1: 데이터 (공통) ──
    with tab1:
        st.subheader("업로드 데이터")
        st.caption(f"유형: {data_type_key}  |  총 {len(df):,}행 x {len(df.columns)}열")
        st.dataframe(df, use_container_width=True)
        if outlier_count > 0:
            status = "제거됨" if remove_outliers else "포함됨 (사이드바에서 제거 가능)"
            st.warning(f"이상치 {outlier_count}개 탐지 — 현재 {status}")

    # ── 탭2: 그래프 (실험 기구) / 분석 결과 (일반) ──
    with tab2:
        # 산점도 + 회귀선 (공통)
        fig = px.scatter(data, x=x_col, y=y_col,
                         title=f"{x_col} vs {y_col}  [{data_type_key}]", opacity=0.7)
        if remove_outliers and outlier_count > 0:
            fig.add_scatter(
                x=data_raw[outlier_mask][x_col],
                y=data_raw[outlier_mask][y_col],
                mode="markers",
                marker=dict(color="gray", symbol="x", size=8),
                name=f"이상치 ({outlier_count}개)"
            )
        fig.add_scatter(
            x=sorted_data[x_col].values, y=y_line,
            mode="lines", name=f"회귀선 ({degree_label[best_degree]})",
            line=dict(color="red", width=2)
        )
        st.plotly_chart(fig, use_container_width=True)

        if is_instrument:
            # 실험 기구: 회귀식과 간단한 안내만 표시
            st.info(
                f"이 데이터는 **{data_type_key}** 유형으로 설정되어 있습니다. "
                "R² 등 통계 지표는 표시하지 않으며, 그래프와 회귀식만 제공합니다."
            )
            st.subheader("회귀식 (선형 기준)")
            st.code(regression_formula)

        else:
            # 일반 실험: 기존 전체 분석 표시
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("최적 모델", degree_label[best_degree])
            col2.metric("R²", f"{r2:.4f}")
            cv_display_ui = (
                f"{best['cv_mean']:.4f} ± {best['cv_std']:.4f}"
                if not np.isnan(best["cv_mean"]) else "N/A"
            )
            col3.metric("교차검증 R²", cv_display_ui)
            col4.metric("RMSE", f"{rmse:.4f}")
            col5.metric("p-value", p_text)
            st.markdown("---")

            st.subheader("모델 비교")
            compare_rows = []
            for d in [1, 2, 3]:
                mr = model_results[d]
                cv_str = (
                    f"{mr['cv_mean']:.4f} ± {mr['cv_std']:.4f}"
                    if not np.isnan(mr["cv_mean"]) else "N/A"
                )
                compare_rows.append({
                    "모델": degree_label[d],
                    "R²": f"{mr['r2']:.4f}",
                    "교차검증 R²": cv_str,
                    "추천": "✅ 최적" if d == best_degree else ""
                })
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
            st.markdown("---")

            st.subheader("통계적 유의성")
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("p-value", p_text)
            col_p2.metric("유의성", "유의함 ✅" if p_value < 0.05 else "유의하지 않음 ❌")
            st.caption("p < 0.05이면 회귀 관계가 통계적으로 유의합니다.")
            if p_value >= 0.05:
                st.warning(
                    "⚠️ p-value가 0.05 이상입니다. 통계적으로 유의하지 않으므로 "
                    "결과 해석 시 주의가 필요합니다."
                )
            elif p_value >= 0.01:
                st.info(
                    "ℹ️ p-value가 0.01~0.05 구간입니다. "
                    "추가 데이터 수집이나 반복 실험을 권장합니다."
                )
            st.markdown("---")

            st.subheader("이상치 탐지")
            col_o1, col_o2 = st.columns(2)
            col_o1.metric("탐지된 이상치", f"{outlier_count}개")
            col_o2.metric("분석에 사용된 데이터", f"{len(data)}개")
            if outlier_count > 0:
                st.caption(
                    f"Z-score {zscore_threshold} 초과 데이터를 이상치로 분류했습니다. "
                    "사이드바에서 제거 여부를 선택할 수 있습니다."
                )
            st.markdown("---")

            st.subheader("추세 분석")
            st.success(trend_text)
            st.subheader("타당성 검토")
            st.info(validity_text)
            st.subheader("해석 및 활용")
            st.write(insight_text)
            st.subheader("회귀식 (선형 기준)")
            st.code(regression_formula)
            st.markdown("---")

            st.subheader("잔차 플롯")
            fig_resid = px.scatter(
                x=y_pred, y=residuals,
                labels={"x": "예측값", "y": "잔차"},
                title="잔차 vs 예측값", opacity=0.7
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_resid, use_container_width=True)

    # ── 탭3: AI 해석 (일반) / 다운로드 (실험 기구) ──
    with tab3:
        if is_instrument:
            # 실험 기구 다운로드
            st.subheader("다운로드")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**결과 데이터 (CSV)**")
                result_df = data.copy()
                result_df["회귀선 예측값"] = y_pred
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    label="CSV 다운로드",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="graph_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_b:
                st.markdown("**그래프 리포트 (PDF)**")
                if st.button("PDF 리포트 생성", use_container_width=True):
                    with st.spinner("PDF 생성 중..."):
                        pdf_buf = generate_pdf()
                    st.download_button(
                        label="PDF 다운로드",
                        data=pdf_buf,
                        file_name="graph_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            # 일반 실험 AI 해석
            st.subheader("AI 자동 해석")
            if st.session_state.ai_interpretation:
                st.markdown(st.session_state.ai_interpretation)
            else:
                st.info(
                    "사이드바의 **'AI 자동 해석 생성'** 버튼을 눌러 주세요.\n\n"
                    "분석 결과를 바탕으로 학술적 해석을 자동으로 작성해 드립니다."
                )

    # ── 탭4: 다운로드 (일반 실험 전용) ──
    if not is_instrument:
        with tab4:
            st.subheader("다운로드")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**결과 데이터 (CSV)**")
                result_df = data.copy()
                result_df["예측값"] = y_pred
                result_df["잔차"] = residuals
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    label="CSV 다운로드",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="regression_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_b:
                st.markdown("**분석 리포트 (PDF)**")
                if st.session_state.ai_interpretation:
                    st.success("AI 해석 포함 리포트 준비 완료")
                else:
                    st.info("AI 해석을 먼저 생성하면 PDF에 포함됩니다.")
                if st.button("PDF 리포트 생성", use_container_width=True):
                    with st.spinner("PDF 생성 중..."):
                        pdf_buf = generate_pdf()
                    st.download_button(
                        label="PDF 다운로드",
                        data=pdf_buf,
                        file_name="analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

except Exception as e:
    st.error(f"오류 발생: {e}")
    with st.expander("상세 오류 보기"):
        st.code(traceback.format_exc())
