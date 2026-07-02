import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

st.set_page_config(page_title="실험 데이터 분석 앱", layout="wide")

st.title("📊 실험 데이터 분석 ")

file = st.file_uploader("CSV 파일 업로드", type="csv")

if file is not None:

    try:
        # -------------------------
        # 인코딩 자동 감지
        # -------------------------
        encodings = ["utf-8", "cp949", "euc-kr", "utf-8-sig"]

        df = None

        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
                st.success(f"파일 인코딩 감지: {enc}")
                break
            except:
                continue

        if df is None:
            st.error("CSV 인코딩을 인식할 수 없습니다.")
            st.stop()

        # -------------------------
        # 데이터 표시
        # -------------------------
        st.subheader("📄 데이터 미리보기")
        st.dataframe(df.head(20))

        # -------------------------
        # 숫자 컬럼 선택
        # -------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("숫자형 컬럼이 2개 이상 필요합니다.")
            st.stop()

        x_col = st.selectbox("X축 선택", numeric_cols)
        y_candidates = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Y축 선택", y_candidates)

        data = df[[x_col, y_col]].dropna()

        if len(data) < 2:
            st.error("유효한 데이터가 부족합니다.")
            st.stop()

        X = data[[x_col]].values
        y = data[y_col].values

        # -------------------------
        # 모델 학습
        # -------------------------
        model = LinearRegression()
        model.fit(X, y)

        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)

        # -------------------------
        # 상관계수
        # -------------------------
        corr = data.corr().iloc[0, 1]

        col1, col2, col3 = st.columns(3)
        col1.metric("📈 R² Score", f"{r2:.3f}")
        col2.metric("📊 상관계수", f"{corr:.3f}")
        col3.metric("📉 기울기", f"{slope:.3f}")

        # -------------------------
        # 그래프
        # -------------------------
        st.subheader("📊 시각화")

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=f"{y_col} vs {x_col}"
        )

        sorted_idx = np.argsort(data[x_col].values)

        fig.add_scatter(
            x=data[x_col].values[sorted_idx],
            y=y_pred[sorted_idx],
            mode="lines",
            name="회귀선",
            line=dict(color="red")
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # 추세 해석
        # -------------------------
        st.subheader("🧠 분석 결과")

        if slope > 0:
            trend = "X가 증가하면 Y도 증가하는 경향"
        elif slope < 0:
            trend = "X가 증가하면 Y는 감소하는 경향"
        else:
            trend = "뚜렷한 관계 없음"

        st.success(trend)

        # 상관관계 해석
        if abs(corr) > 0.9:
            st.info("👉 매우 강한 선형 관계")
        elif abs(corr) > 0.7:
            st.info("👉 강한 선형 관계")
        elif abs(corr) > 0.4:
            st.warning("👉 약한 선형 관계")
        else:
            st.error("👉 거의 관계 없음")

        # -------------------------
        # 회귀식
        # -------------------------
        st.subheader("📐 회귀식")

        st.code(f"{y_col} = {slope:.4f} × {x_col} + {intercept:.4f}")

        # -------------------------
        # 예측 테이블
        # -------------------------
        st.subheader("📋 실제 vs 예측")

        result_df = data.copy()
        result_df["예측값"] = y_pred
        st.dataframe(result_df.head(20))

        # -------------------------
        # 예측 기능
        # -------------------------
        st.subheader("🔮 값 입력 예측")

        input_value = st.number_input(f"{x_col} 값 입력")

        if st.button("예측 실행"):
            prediction = model.predict([[input_value]])
            st.success(f"예측된 {y_col}: {prediction[0]:.4f}")

    except Exception as e:
        st.error(f"오류 발생: {e}")
