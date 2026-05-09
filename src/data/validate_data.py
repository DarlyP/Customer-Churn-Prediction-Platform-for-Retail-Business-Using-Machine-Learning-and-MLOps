from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from jinja2 import Template

from src.utils.config import load_config


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load interim dataset if available.
    If interim data does not exist, fallback to raw data.
    """
    interim_path = Path(config["data"]["interim_path"])
    raw_path = Path(config["data"]["raw_path"])

    if interim_path.exists():
        data_path = interim_path
    elif raw_path.exists():
        data_path = raw_path
    else:
        raise FileNotFoundError(
            f"Neither interim nor raw data found: {interim_path}, {raw_path}"
        )

    df = pd.read_csv(data_path)
    return df


def create_gx_batch(df: pd.DataFrame):
    """
    Create Great Expectations batch from pandas DataFrame.
    """
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("pandas_runtime")
    data_asset = data_source.add_dataframe_asset(name="retail_churn_dataframe")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "retail_churn_batch"
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    return batch


def run_expectation(batch, expectation, expectation_name: str) -> dict[str, Any]:
    """
    Run a single Great Expectations expectation and return a simplified result.
    """
    result = batch.validate(expectation)

    return {
        "expectation": expectation_name,
        "success": bool(result.success),
        "details": result.to_json_dict(),
    }


def validate_required_columns(
    batch,
    required_columns: list[str],
    results: list[dict[str, Any]],
) -> None:
    """
    Validate that required columns exist.
    """
    for column in required_columns:
        expectation = gx.expectations.ExpectColumnToExist(column=column)

        results.append(
            run_expectation(
                batch=batch,
                expectation=expectation,
                expectation_name=f"Column exists: {column}",
            )
        )


def validate_id_column_not_null(
    batch,
    id_column: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Validate that ID column does not contain null values.
    """
    expectation = gx.expectations.ExpectColumnValuesToNotBeNull(column=id_column)

    results.append(
        run_expectation(
            batch=batch,
            expectation=expectation,
            expectation_name=f"ID column not null: {id_column}",
        )
    )


def validate_non_negative_columns(
    batch,
    non_negative_columns: list[str],
    available_columns: list[str],
    results: list[dict[str, Any]],
) -> None:
    """
    Validate that selected numerical columns contain non-negative values.
    """
    for column in non_negative_columns:
        if column not in available_columns:
            results.append(
                {
                    "expectation": f"Non-negative column skipped: {column}",
                    "success": False,
                    "details": {
                        "error": f"Column '{column}' not found in dataset."
                    },
                }
            )
            continue

        expectation = gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=0,
        )

        results.append(
            run_expectation(
                batch=batch,
                expectation=expectation,
                expectation_name=f"Non-negative values: {column}",
            )
        )


def validate_categorical_columns(
    batch,
    categorical_columns: dict[str, list[Any]],
    available_columns: list[str],
    results: list[dict[str, Any]],
) -> None:
    """
    Validate that categorical columns only contain allowed values.
    """
    for column, allowed_values in categorical_columns.items():
        if column not in available_columns:
            results.append(
                {
                    "expectation": f"Categorical column skipped: {column}",
                    "success": False,
                    "details": {
                        "error": f"Column '{column}' not found in dataset."
                    },
                }
            )
            continue

        expectation = gx.expectations.ExpectColumnValuesToBeInSet(
            column=column,
            value_set=allowed_values,
        )

        results.append(
            run_expectation(
                batch=batch,
                expectation=expectation,
                expectation_name=f"Allowed category values: {column}",
            )
        )


def validate_missing_ratio(
    df: pd.DataFrame,
    max_missing_ratio: float,
    results: list[dict[str, Any]],
) -> None:
    """
    Validate missing ratio manually because this is a dataset-wide rule.
    """
    missing_ratio = df.isna().mean()
    failed_columns = missing_ratio[missing_ratio > max_missing_ratio]

    success = failed_columns.empty

    results.append(
        {
            "expectation": f"Missing ratio <= {max_missing_ratio}",
            "success": success,
            "details": {
                "max_missing_ratio": max_missing_ratio,
                "failed_columns": failed_columns.to_dict(),
                "all_missing_ratios": missing_ratio.to_dict(),
            },
        }
    )


def generate_html_report(
    results: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Generate simple HTML validation report.
    """
    total_checks = len(results)
    passed_checks = sum(1 for result in results if result["success"])
    failed_checks = total_checks - passed_checks

    html_template = Template(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Data Validation Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #222;
                }
                h1 {
                    color: #111;
                }
                .summary {
                    padding: 16px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    margin-bottom: 24px;
                    background: #f9f9f9;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 10px;
                    text-align: left;
                    vertical-align: top;
                }
                th {
                    background: #f2f2f2;
                }
                .passed {
                    color: green;
                    font-weight: bold;
                }
                .failed {
                    color: red;
                    font-weight: bold;
                }
                pre {
                    white-space: pre-wrap;
                    max-height: 300px;
                    overflow-y: auto;
                    background: #f7f7f7;
                    padding: 10px;
                    border-radius: 6px;
                }
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>

            <div class="summary">
                <p><strong>Total checks:</strong> {{ total_checks }}</p>
                <p><strong>Passed:</strong> {{ passed_checks }}</p>
                <p><strong>Failed:</strong> {{ failed_checks }}</p>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>No</th>
                        <th>Expectation</th>
                        <th>Status</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in results %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ result.expectation }}</td>
                        <td>
                            {% if result.success %}
                                <span class="passed">PASSED</span>
                            {% else %}
                                <span class="failed">FAILED</span>
                            {% endif %}
                        </td>
                        <td>
                            <pre>{{ result.details }}</pre>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    html = html_template.render(
        results=results,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )

    output_file.write_text(html, encoding="utf-8")

    print(f"Validation report saved to: {output_path}")


def main() -> None:
    config = load_config()

    validation_config = config["validation"]

    required_columns = validation_config.get("required_columns", [])
    id_column = validation_config.get("id_column")
    non_negative_columns = validation_config.get("non_negative_columns", [])
    categorical_columns = validation_config.get("categorical_columns", {})
    max_missing_ratio = validation_config.get("max_missing_ratio", 0.30)
    report_path = validation_config["report_path"]

    df = load_dataset(config)
    batch = create_gx_batch(df)

    available_columns = df.columns.tolist()
    results: list[dict[str, Any]] = []

    validate_required_columns(
        batch=batch,
        required_columns=required_columns,
        results=results,
    )

    if id_column:
        validate_id_column_not_null(
            batch=batch,
            id_column=id_column,
            results=results,
        )

    validate_non_negative_columns(
        batch=batch,
        non_negative_columns=non_negative_columns,
        available_columns=available_columns,
        results=results,
    )

    validate_categorical_columns(
        batch=batch,
        categorical_columns=categorical_columns,
        available_columns=available_columns,
        results=results,
    )

    validate_missing_ratio(
        df=df,
        max_missing_ratio=max_missing_ratio,
        results=results,
    )

    generate_html_report(
        results=results,
        output_path=report_path,
    )

    failed_checks = [result for result in results if not result["success"]]

    if failed_checks:
        raise ValueError(
            f"Data validation failed. Failed checks: {len(failed_checks)}. "
            f"See report: {report_path}"
        )

    print("Data validation passed.")


if __name__ == "__main__":
    main()