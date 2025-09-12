# --------------------------------------------------
# 0. 라이브러리 임포트

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
os.chdir("C:/Users/semi0/Desktop/공모전/제1회 데이터 분석 경진 대회")
# --------------------------------------------------
# 1. 여러 CSV 파일 읽어서 MEGA_RAW_DATA 만들기
csv_folder = "RAW DATA"  # 네 폴더 경로로 수정

csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

mega_raw_list = []

for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="utf-8", delimiter="|")
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding="cp949", delimiter="|")
    mega_raw_list.append(df)

mega_raw_data = pd.concat(mega_raw_list, axis=0, ignore_index=True)

# --------------------------------------------------
# 2. MEGA_RAW 데이터 컬럼 rename 및 YEAR, MONTH 추가
column_mapping = {
    "TRDAR_NO": "AREA_ID",
    "DATA_STRD_YM": "DATA_YM",
    "TRDAR_ISE_TOTL_BILD_CNT": "TOTAL_BIDG",
    "TRDAR_ISE_1KIND_NGHBRHD_LVLH_FCLTY_CNT": "FAC_NEIGH_1",
    "TRDAR_ISE_2KIND_NGHBRHD_LVLH_FCLTY_CNT": "FAC_NEIGH_2",
    "TRDAR_ISE_CLTUR_AND_MTNG_FCLTY_CNT": "FAC_CULT_MTG",
    "TRDAR_ISE_RLGN_LVLH_FCLTY_CNT": "FAC_RELG",
    "TRDAR_ISE_NTSL_FCLTY_CNT": "FAC_RETAIL",
    "TRDAR_ISE_MLFLT_CNT": "FAC_MEDI",
    "TRDAR_ISE_OVNTY_FCLTY_CNT": "FAC_YOSE",
    "TRDAR_ISE_NVTT_FCLTY_CNT": "FAC_TRAIN",
    "TRDAR_ISE_EXRCS_FCLTY_CNT": "FAC_SPORT",
    "TRDAR_ISE_ACMT_CNT": "FAC_STAY",
    "TRDAR_ISE_WAF_FCLTY_CNT": "FAC_LEISURE",
    "TRDAR_ISE_GAS_TOTL_USQNT": "TOTAL_GAS",
    "CMRC_PUP_BULD_GAS_USQNT": "CMRC_GAS",
    "TRDAR_ISE_FTRM_TOTL_USQNT": "TOTAL_ELEC"
}
mega_raw_data = mega_raw_data[list(column_mapping.keys())].rename(columns=column_mapping)

mega_raw_data["YEAR"] = mega_raw_data["DATA_YM"] // 100
mega_raw_data["MONTH"] = mega_raw_data["DATA_YM"] % 100

# --------------------------------------------------
# 3. TRAIN / TEST 데이터 불러오기
train_data = pd.read_csv("DATA/TRAIN_DATA.csv", encoding="cp949")
test_data = pd.read_csv("DATA/TEST_DATA.csv", encoding="cp949")

# --------------------------------------------------
# 4. 원본 컬럼 기억하기
original_train_cols = train_data.columns.tolist()
original_test_cols = test_data.columns.tolist()

# --------------------------------------------------# --------------------------------------------------
# 5. mega_raw_data를 AREA_ID + DATA_YM 단위로 집계 (중복 제거용)

# 1. 숫자형 컬럼만 집계 대상으로 사용
group_cols = ["AREA_ID", "DATA_YM"]
agg_cols = [col for col in mega_raw_data.columns if col not in group_cols]

# 2. 평균값으로 집계
mega_grouped = mega_raw_data.groupby(group_cols)[agg_cols].mean().reset_index()

# 3. train/test에 안전하게 merge (이제 1:1 매칭됨)
train_data_final = train_data.merge(mega_grouped, on=["AREA_ID", "DATA_YM"], how="left", suffixes=('', '_mega'))
test_data_final = test_data.merge(mega_grouped, on=["AREA_ID", "DATA_YM"], how="left", suffixes=('', '_mega'))

# 4. merge 후 행 수 체크
print("✅ 원래 test 행 수:", len(test_data))
print("✅ merge 후 test_data_final 행 수:", len(test_data_final))


# --------------------------------------------------
# 6. 파생변수 추가 (FAC_SUM, OTHER_BIDG, OTHER_BIDG_RATIO)
fac_cols = ["FAC_NEIGH_1", "FAC_NEIGH_2", "FAC_CULT_MTG", "FAC_RELG",
            "FAC_RETAIL", "FAC_MEDI", "FAC_YOSE", "FAC_TRAIN",
            "FAC_SPORT", "FAC_STAY", "FAC_LEISURE"]

for df in [train_data_final, test_data_final]:
    df['FAC_SUM'] = df[fac_cols].sum(axis=1)
    df['OTHER_BIDG'] = df['TOTAL_BIDG'] - df['FAC_SUM']
    df['OTHER_BIDG'] = df['OTHER_BIDG'].clip(lower=0)
    df['OTHER_BIDG_RATIO'] = df['OTHER_BIDG'] / df['TOTAL_BIDG']

for df in [train_data_final, test_data_final]:
    df['IS_WINTER'] = df['MONTH'].isin([12, 1, 2]).astype(int)
    df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)
# --------------------------------------------------
# 7. 학습 데이터 준비
drop_cols_train = ["DATA_YM", "AREA_ID", "TOTAL_ELEC", "TOTAL_ELEC_mega"]
drop_cols_test = ["DATA_YM", "AREA_ID", "TOTAL_ELEC", "TOTAL_ELEC_mega"]


X = train_data_final.drop(columns=drop_cols_train, axis=1, errors="ignore")
X = X.select_dtypes(include=["number"])  # 숫자형 데이터만 남기기

y = train_data_final["TOTAL_ELEC"].values

X_test = test_data_final.drop(columns=drop_cols_test, axis=1, errors="ignore")
X_test = X_test.select_dtypes(include=["number"])  # 숫자형 데이터만 남기기


# 7.5 파생변수 고급 추가
def get_existing_col(df, col_candidates):
    return next((col for col in col_candidates if col in df.columns), None)

for df in [train_data_final, test_data_final]:
    elec_col = get_existing_col(df, ['TOTAL_ELEC_mega', 'TOTAL_ELEC'])
    gas_col = get_existing_col(df, ['TOTAL_GAS_mega', 'TOTAL_GAS'])

    if elec_col and gas_col:
        df['ELEC_PER_BLDG'] = df[elec_col] / df['TOTAL_BIDG']
        df['GAS_PER_BLDG'] = df[gas_col] / df['TOTAL_BIDG']
        df['ELEC_GAS_RATIO'] = df[elec_col] / (df[gas_col] + 1)
# --------------------------------------------------
# 7.6 파생변수 반영된 X, X_test 재생성 + 정리

drop_cols_train = ["DATA_YM", "AREA_ID", "TOTAL_ELEC", "TOTAL_ELEC_mega"]
drop_cols_test = ["DATA_YM", "AREA_ID", "TOTAL_ELEC", "TOTAL_ELEC_mega"]

X = train_data_final.drop(columns=drop_cols_train, axis=1, errors="ignore").select_dtypes(include=["number"])
X_test = test_data_final.drop(columns=drop_cols_test, axis=1, errors="ignore").select_dtypes(include=["number"])

# 결측치 처리 (NaN -> 0)
X = X.fillna(0)
X_test = X_test.fillna(0)

# 컬럼 정렬 통일 (X_train, X_test 동일하게)
X, X_test = X.align(X_test, join="inner", axis=1)

# --------------------------------------------------
# 8. train/valid 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------

# 9. 모델 학습 (XGBoost)
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train, y_train)
# --------------------------------------------------
# 10. 검증 성능 평가
y_pred_valid = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred_valid, squared=False)
print("Validation RMSE:", rmse)

# --------------------------------------------------
# 11. 테스트 데이터 예측
test_pred = model.predict(X_test)

# --------------------------------------------------
# 12. 제출 파일 생성
# (1) 불필요한 파생변수 삭제
drop_cols_generated = ['FAC_SUM', 'OTHER_BIDG', 'OTHER_BIDG_RATIO', 'YEAR', 'MONTH']

submission = test_data_final.drop(columns=drop_cols_generated, errors='ignore')

submission = pd.DataFrame({"y_pred": test_pred})
submission.to_csv("submission_final.csv", index=False, encoding="utf-8-sig")
print(" 최종 제출파일 저장 완료! (y_pred 컬럼만 포함)")

