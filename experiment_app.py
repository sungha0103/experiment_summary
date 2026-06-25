import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

st.set_page_config(page_title="실험 데이터 자동 분석 앱", layout="wide")

st.title("실험 데이터 자동 분석 앱")

file = st.file_uploader("CSV 파일 업로드", type="csv")

if file is not None:

    try:
        # -------------------------
        # CSV 인코딩 자동 감지
        # -------------------------
        encodings = [
            "utf-8",
            "cp949",
            "euc-kr",
            "utf-8-sig"
        ]

        df = None

        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
                st.success(f"파일 인코딩 감지: {enc}")
                break
            except Exception:
                continue

        if df is None:
            st.error("CSV 파일 인코딩을 인식할 수 없습니다.")
            st.stop()

        st.subheader("업로드 데이터")
        st.dataframe(df)

        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("숫자형 컬럼이 최소 2개 이상 필요합니다.")
            st.stop()

        x_col = st.selectbox("X축 선택", numeric_cols)

        y_candidates = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Y축 선택", y_candidates)

        # 결측치 제거
        data = df[[x_col, y_col]].dropna()

        if len(data) < 2:
            st.error("유효한 데이터가 2개 이상 필요합니다.")
            st.stop()

        X = data[[x_col]].values
        y = data[y_col].values

        # -------------------------
        # 선형 회귀
        # -------------------------
        model = LinearRegression()
        model.fit(X, y)

        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        y_pred = model.predict(X)

        try:
            r2 = r2_score(y, y_pred)
        except:
            r2 = np.nan

        # -------------------------
        # 그래프
        # -------------------------
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=f"{x_col} vs {y_col}"
        )

        sorted_idx = np.argsort(data[x_col].values)

        fig.add_scatter(
            x=data[x_col].values[sorted_idx],
            y=y_pred[sorted_idx],
            mode="lines",
            name="회귀선"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # -------------------------
        # 추세 분석
        # -------------------------
        st.subheader("📈 1. 추세 분석")

        eps = 1e-6

        if slope > eps:
            trend = "X가 증가할수록 Y도 증가하는 경향"
        elif slope < -eps:
            trend = "X가 증가할수록 Y는 감소하는 경향"
        else:
            trend = "뚜렷한 변화가 관찰되지 않음"

        st.write(f"기울기 : {slope:.4f}")
        st.write(f"절편 : {intercept:.4f}")
        st.success(trend)

        st.markdown("---")

        # -------------------------
        # 타당성 검토
        # -------------------------
        st.subheader("📊 2. 회귀 기반 타당성 검토")

        st.write(f"R² : {r2:.4f}")

        if r2 < 0:
            validity = "평균값 예측보다 성능이 낮습니다."
        elif r2 < 0.7:
            validity = "선형 관계가 약합니다."
        elif r2 < 0.9:
            validity = "상당한 선형 관계가 존재합니다."
        else:
            validity = "매우 강한 선형 관계가 존재합니다."

        st.info(validity)

        st.markdown("---")

        # -------------------------
        # 데이터 해석
        # -------------------------
        st.subheader("🧠 3. 데이터 해석 및 활용")

        if r2 >= 0.9 and slope > 0:
            insight = "강한 비례 관계를 가지며 예측 모델로 활용 가능합니다."

        elif r2 >= 0.9 and slope < 0:
            insight = "강한 반비례 관계를 가지며 감소 모델로 해석 가능합니다."

        elif r2 < 0.7:
            insight = "비선형 관계 또는 외부 요인의 영향을 고려해야 합니다."

        else:
            insight = "일정 수준의 경향성을 가지며 추가 분석이 권장됩니다."

        st.write(insight)

        st.markdown("---")

        # -------------------------
        # 회귀식
        # -------------------------
        st.subheader("📐 회귀식")

        regression_formula = (
            f"{y_col} = {slope:.4f} × {x_col} + {intercept:.4f}"
        )

        st.code(regression_formula)

    except Exception as e:
        st.error(f"오류 발생: {e}")

