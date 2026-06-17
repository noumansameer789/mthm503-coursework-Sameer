import click
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import subprocess
import sys
from pathlib import Path
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[3]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
# Now we can import any project-specific utilities if needed
from course.intro.pipeline_functions import (tyler_viglen)  # noqa: E402


@click.command()
@click.option('--out-dir', type=click.Path(), default='artefacts/intro',
              help='Output directory')
@click.option('--template', type=click.Path(), default='reports/templates/intro.md',
              help='Output directory')
@click.option('--report-file', type=click.Path(), default='reports/intro.html',
              help='Output directory')
def main(out_dir, template, report_file):
    # 1. Setup
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    click.echo("Generating and filtering data...")
    df = tyler_viglen()
    # 2. Data Filtering
    # TODO: Remove this logic to course.intro.pipeline_functions.py
    #       And call the function filter_data(df, year) instead
    df_filtered = df[df['Year'] < 2015]
    df_filtered.to_csv(p / 'filtered_data.csv', index=False)
    # 3. Analysis (Correlation)
    click.echo("Calculating correlation...")
    # TODO: Remove this logic to course.intro.pipeline_functions.py
    #       And call the function calculate_correlation(df, x1, x2) instead
    corr_coef, p_val = pearsonr(df['Kerosene'], df['DivorceRate'])
    with open(p / 'correlation.txt', 'w') as f:
        f.write(f"Correlation: {corr_coef:.3f}\nP-value: {p_val:.3f}\n")

    # 4. Modeling (Regression)
    click.echo("Fitting regression...")
    # TODO: Remove this logic to course.intro.pipeline_functions.py
    #       And call the function fit_regression(df, x_name, y_name) instead
    X = sm.add_constant(df['Kerosene'])
    Y = df['DivorceRate']
    model = sm.OLS(Y, X).fit()
    with open(p / 'regression_summary.txt', 'w') as f:
        f.write(model.summary().as_text())

    # 5. Visualization (Matplotlib replacement)
    click.echo("Generating plot...")
    # TODO: Remove this logic to course.intro.pipeline_functions.py
    #       And call the function plot_scatter(df, x_name, y_name) instead
    plt.figure(figsize=(8, 6))
    plt.scatter(df_filtered['Kerosene'], df_filtered['DivorceRate'], color='teal', alpha=0.7)
    plt.title('Spurious Correlation: Kerosene vs Divorce Rate')
    plt.xlabel('Kerosene (consumption)')
    plt.ylabel('Divorce Rate (per 1000)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(p / 'scatterplot.png', dpi=150)
    plt.close()
    with open(template, 'r') as f:
        content = f.read()
    with open(p / 'correlation.txt', 'r') as c:
        corr_report = c.read()
    content = content.replace('{{CORRELATION}}', corr_report)
    with open(p / 'regression_summary.txt', 'r') as r:
        reg_report = r.read()
    content = content.replace('{{REGRESSION_SUMMARY}}', reg_report)
    temp_md = Path("temp_intro_report.md")
    temp_md.write_text(content)
    # 6. Collate report
    click.echo("Generating report...")
    try:
        subprocess.run([
            'pandoc',
            str(temp_md),
            '-o', str(report_file),
            '--to', 'html5',
            '--standalone',
            '--embed-resources'
            ], check=True)

        click.echo(f"Pipeline complete. Outputs saved to {out_dir}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Pandoc failed {e}")
    finally:
        if temp_md.exists():
            temp_md.unlink()
    click.echo("Report written to " + str(report_file))


if __name__ == "__main__":
    main()
