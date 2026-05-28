from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


# 这些列中的缺失值在数据说明里通常表示“没有该设施”，不是普通未知值。
NONE_COLUMNS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

# 这些数值列没有对应设施时可以自然地补 0。
ZERO_COLUMNS = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
]

# 这些数字编码本质上是类别，不应该按连续数值建模。
CATEGORICAL_NUMERIC_COLUMNS = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]


def add_house_features(data: pd.DataFrame) -> pd.DataFrame:
    """基于原始房屋字段构造面积、年龄、设施状态等领域特征。"""
    data = data.copy()

    # 面积、浴室、门廊等聚合特征可以让线性模型捕捉更强的总量信号。
    data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    data["TotalBathrooms"] = (
        data["FullBath"]
        + 0.5 * data["HalfBath"]
        + data["BsmtFullBath"]
        + 0.5 * data["BsmtHalfBath"]
    )
    data["TotalPorchSF"] = (
        data["OpenPorchSF"]
        + data["EnclosedPorch"]
        + data["3SsnPorch"]
        + data["ScreenPorch"]
        + data["WoodDeckSF"]
    )

    # 年龄类特征比原始年份更接近购房者感知。
    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
    data["GarageAge"] = data["YrSold"] - data["GarageYrBlt"]

    # 是否存在某类设施通常是强信号，用 0/1 特征显式表达。
    data["IsRemodeled"] = (data["YearBuilt"] != data["YearRemodAdd"]).astype(int)
    data["HasPool"] = (data["PoolArea"] > 0).astype(int)
    data["HasGarage"] = (data["GarageArea"] > 0).astype(int)
    data["HasBasement"] = (data["TotalBsmtSF"] > 0).astype(int)
    data["HasFireplace"] = (data["Fireplaces"] > 0).astype(int)
    return data


def prepare_raw_features(data: pd.DataFrame, *, has_target: bool) -> pd.DataFrame:
    """删除非特征列，并完成进入 sklearn Pipeline 前的数据集专属清洗。"""
    drop_columns = ["Id"]
    if has_target:
        drop_columns.append("SalePrice")

    features = data.drop(columns=drop_columns).copy()

    for column in NONE_COLUMNS:
        if column in features.columns:
            features[column] = features[column].fillna("None")

    for column in ZERO_COLUMNS:
        if column in features.columns:
            features[column] = features[column].fillna(0)

    features = add_house_features(features)

    for column in CATEGORICAL_NUMERIC_COLUMNS:
        if column in features.columns:
            features[column] = features[column].astype("object")

    return features


def split_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """根据 pandas dtype 将特征列拆分为数值列和类别列。"""
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_columns, categorical_columns


def find_skewed_numeric_columns(
    features: pd.DataFrame, numeric_columns: list[str], threshold: float = 0.75
) -> list[str]:
    """找出偏度较高且非负的数值列，用于后续 log1p 变换。"""
    skew = features[numeric_columns].skew(numeric_only=True).abs()
    return [
        column
        for column in skew[skew > threshold].index
        if features[column].min(skipna=True) >= 0
    ]


def safe_log1p(values: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """对输入先截断负值再做 log1p，避免测试集异常值产生无穷值。"""
    # 测试集偶尔会让年龄类派生特征变成负数，先截断可避免 log1p 产生 inf。
    return np.log1p(np.clip(values, 0, None))


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """创建 sklearn 预处理器，分别处理偏态数值列、普通数值列和类别列。"""
    numeric_columns, categorical_columns = split_feature_types(features)
    skewed_numeric_columns = find_skewed_numeric_columns(features, numeric_columns)
    regular_numeric_columns = [
        column for column in numeric_columns if column not in skewed_numeric_columns
    ]

    # 数值列：偏态严重且非负的列先 log1p，再补中位数、标准化。
    skewed_numeric_pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(safe_log1p, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    regular_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 类别列：剩余缺失值统一视作 Missing，并用 handle_unknown 支持测试集新类别。
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("skewed_numeric", skewed_numeric_pipeline, skewed_numeric_columns),
            ("regular_numeric", regular_numeric_pipeline, regular_numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )


def make_model(features: pd.DataFrame, alpha: float) -> Pipeline:
    """创建完整建模 Pipeline：预处理器接 Ridge 回归模型。"""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(features)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def remove_known_outliers(
    x: pd.DataFrame, y_log: np.ndarray, original_train: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """移除比赛中常见的两个高面积低售价异常训练样本。"""
    # Kaggle 讨论中常见的两个异常点：居住面积很大但售价异常低。
    mask = ~(
        (original_train["GrLivArea"] > 4000)
        & (original_train["SalePrice"] < 300000)
    )
    removed = int((~mask).sum())
    return x.loc[mask].reset_index(drop=True), y_log[mask.to_numpy()], removed


def cross_validate(
    x: pd.DataFrame,
    y_log: np.ndarray,
    alphas: list[float],
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """对候选 Ridge alpha 执行 K 折验证，并返回按 RMSLE 排序的结果表。"""
    rows = []
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for alpha in alphas:
        base_model = make_model(x, alpha=alpha)
        fold_scores = []

        for train_index, valid_index in cv.split(x):
            model = clone(base_model)
            x_train = x.iloc[train_index]
            x_valid = x.iloc[valid_index]
            y_train = y_log[train_index]
            y_valid = y_log[valid_index]

            model.fit(x_train, y_train)
            pred_valid = model.predict(x_valid)
            fold_scores.append(root_mean_squared_error(y_valid, pred_valid))

        rows.append(
            {
                "alpha": alpha,
                "mean_rmsle": float(np.mean(fold_scores)),
                "std_rmsle": float(np.std(fold_scores)),
                **{f"fold_{i + 1}": score for i, score in enumerate(fold_scores)},
            }
        )

    return pd.DataFrame(rows).sort_values("mean_rmsle").reset_index(drop=True)


def positive_price_from_log_prediction(pred_log: np.ndarray) -> np.ndarray:
    """将 log 空间预测还原为房价，并确保预测值为正数。"""
    return np.maximum(np.expm1(pred_log), 1.0)


def save_coefficients(path: Path, model: Pipeline) -> None:
    """导出最终 Ridge 模型的特征名、系数和系数绝对值。"""
    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["model"]

    coefficient_rows = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient": ridge.coef_,
            "abs_coefficient": np.abs(ridge.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coefficient_rows.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，包括数据目录、输出目录、折数和随机种子。"""
    parser = argparse.ArgumentParser(description="Train a Kaggle house price model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Keep the two common high-GrLivArea low-price training outliers.",
    )
    return parser.parse_args()


def main() -> None:
    """运行完整训练流程，并生成验证结果、系数文件和 Kaggle 提交文件。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")

    train_x = prepare_raw_features(train, has_target=True)
    test_x = prepare_raw_features(test, has_target=False)
    test_ids = test["Id"].copy()
    y_log = np.log1p(train["SalePrice"].to_numpy(dtype=np.float64))

    outlier_count = 0
    if not args.keep_outliers:
        train_x, y_log, outlier_count = remove_known_outliers(train_x, y_log, train)

    alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0, 100.0, 200.0, 400.0, 800.0]
    cv_results = cross_validate(
        train_x, y_log, alphas=alphas, n_splits=args.folds, seed=args.seed
    )
    cv_results.to_csv(args.output_dir / "cv_results.csv", index=False)

    best_alpha = float(cv_results.loc[0, "alpha"])
    final_model = make_model(train_x, alpha=best_alpha)
    final_model.fit(train_x, y_log)
    pred_log = final_model.predict(test_x)
    predictions = positive_price_from_log_prediction(pred_log)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    save_coefficients(args.output_dir / "ridge_coefficients.csv", final_model)

    print(f"train rows: {len(train_x)}")
    print(f"test rows: {len(test_x)}")
    print(f"raw features: {train_x.shape[1]}")
    print(
        "model features: "
        f"{len(final_model.named_steps['preprocessor'].get_feature_names_out())}"
    )
    print(f"removed outliers: {outlier_count}")
    print(f"best alpha: {best_alpha:g}")
    print(f"best cv rmsle: {cv_results.loc[0, 'mean_rmsle']:.5f}")
    print(f"submission: {args.output_dir / 'submission.csv'}")
    print(f"cv results: {args.output_dir / 'cv_results.csv'}")


if __name__ == "__main__":
    main()
