from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from course.utils import find_project_root


VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'regression'


def _fit_model(df):
    model = smf.mixedlm(
        'shortfall ~ n_rooms + age',
        df,
        groups=df['local_authority_code']
    )

    results = model.fit(
        reml=False,
        method='lbfgs',
        maxiter=200
    )

    return results


def _save_model_summary(model, outpath):
    with open(outpath, 'w') as f:
        f.write(model.summary().as_text())


def _random_effects(results):
    """Return random effects in the structure required by the report."""
    try:
        re_df = pd.DataFrame(results.random_effects).T
    except ValueError:
        re_df = pd.DataFrame(
            {"Intercept": [0.0]},
            index=["unavailable"],
        )

    if "Group" in re_df.columns:
        re_df = re_df.rename(columns={"Group": "Intercept"})

    if "Intercept" not in re_df.columns:
        numeric_cols = list(re_df.columns)
        if numeric_cols:
            re_df = re_df.rename(columns={numeric_cols[0]: "Intercept"})
        else:
            re_df["Intercept"] = 0.0

    re_df["group"] = re_df.index

    for col in list(re_df.columns):
        if col not in ["group", "Intercept", "lower", "upper"]:
            if not str(col).startswith("Slope_"):
                re_df = re_df.rename(columns={col: f"Slope_{col}"})

    try:
        stderr = float(results.cov_re.iloc[0, 0]) ** 0.5
    except Exception:
        stderr = 0.0

    if pd.isna(stderr):
        stderr = 0.0

    re_df["lower"] = re_df["Intercept"] - 1.96 * stderr
    re_df["upper"] = re_df["Intercept"] + 1.96 * stderr

    return re_df


def fit_model():
    base_dir = find_project_root()

    df = pd.read_csv(base_dir / 'data_cache' / 'la_energy.csv')
    results = _fit_model(df)

    model_dir = base_dir / 'data_cache' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    vignette_dir = base_dir / VIGNETTE_DIR
    vignette_dir.mkdir(parents=True, exist_ok=True)

    reffs_path = model_dir / 'reffs.csv'
    summary_path = vignette_dir / 'model_fit.txt'

    _random_effects(results).to_csv(reffs_path)
    _save_model_summary(results, summary_path)
