"""
Several utility functions are reused from Harvard's spatial data lab repository for Geospatial Analytics Extension.
https://github.com/spatial-data-lab/knime-geospatial-extension/blob/main/knime_extension/src/util/knime_utils.py
"""

import knime.extension as knext
import pandas as pd
from typing import Callable
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

############################################
# Timestamp column selection helper
############################################

# Strings of IDs of date/time value factories
ZONED_DATE_TIME_ZONE_VALUE = "org.knime.core.data.v2.time.ZonedDateTimeValueFactory2"
LOCAL_TIME_VALUE = "org.knime.core.data.v2.time.LocalTimeValueFactory"
LOCAL_DATE_VALUE = "org.knime.core.data.v2.time.LocalDateValueFactory"
LOCAL_DATE_TIME_VALUE = "org.knime.core.data.v2.time.LocalDateTimeValueFactory"


DEF_ZONED_DATE_LABEL = "ZonedDateTimeValueFactory2"
DEF_DATE_LABEL = "LocalDateValueFactory"
DEF_TIME_LABEL = "LocalTimeValueFactory"
DEF_DATE_TIME_LABEL = "LocalDateTimeValueFactory"

# Timestamp formats
ZONED_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"


def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return column.ktype == knext.double() or column.ktype == knext.int32() or column.ktype == knext.int64()


def is_string(column: knext.Column) -> bool:
    """
    Checks if column is a string type.
    @return: True if Column is a string
    """
    return column.ktype == knext.string()


def is_zoned_datetime(column: knext.Column) -> bool:
    """
    Checks if date&time column contains has the timezone or not.
    @return: True if selected date&time column has time zone
    """
    return __is_type_x(column, ZONED_DATE_TIME_ZONE_VALUE)


def is_datetime(column: knext.Column) -> bool:
    """
    Checks if a column is of type Date&Time.
    @return: True if selected column is of type date&time
    """
    return __is_type_x(column, LOCAL_DATE_TIME_VALUE)


def is_time(column: knext.Column) -> bool:
    """
    Checks if a column is of type Time only.
    @return: True if selected column has only time.
    """
    return __is_type_x(column, LOCAL_TIME_VALUE)


def is_date(column: knext.Column) -> bool:
    """
    Checks if a column is of type date only.
    @return: True if selected column has date only.
    """
    return __is_type_x(column, LOCAL_DATE_VALUE)


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def is_type_timestamp(column: knext.Column):
    """
    This function checks on all the supported timestamp columns supported in KNIME.
    Note that legacy date&time types are not supported.
    @return: True if date&time column is compatible with the respective logical types supported in KNIME.
    """

    return boolean_or(is_time, is_date, is_datetime, is_zoned_datetime)(column)


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type whereas type can be :
    DateTime, Date, Time, ZonedDateTime
    @return: True if column type is of type timestamp
    """

    return isinstance(column.ktype, knext.LogicalType) and type in column.ktype.logical_type


############################################
# General Helper Class
############################################


def column_exists_or_preset(
    context: knext.ConfigurationContext,
    column: str,
    schema: knext.Schema,
    func: Callable[[knext.Column], bool] = None,
    none_msg: str = "No compatible column found in input table",
) -> str:
    """
    Checks that the given column is not None and exists in the given schema. If none is selected it returns the
    first column that is compatible with the provided function. If none is compatible it throws an exception.
    """
    if column is None:
        for c in schema:
            if func(c):
                context.set_warning(f"Preset column to: {c.name}")
                return c.name
        raise knext.InvalidParametersError(none_msg)
    __check_col_and_type(column, schema, func)
    return column


def __check_col_and_type(
    column: str,
    schema: knext.Schema,
    check_type: Callable[[knext.Column], bool] = None,
) -> None:
    """
    Checks that the given column exists in the given schema and that it matches the given type_check function.
    """
    # Check that the column exists in the schema and that it has a compatible type
    try:
        existing_column = schema[column]
        if check_type is not None and not check_type(existing_column):
            raise knext.InvalidParametersError(f"Column '{str(column)}' has incompatible data type")
    except IndexError:
        raise knext.InvalidParametersError(f"Column '{str(column)}' not available in input table")


############################################
# Generic pandas dataframe/series helper function
############################################


def check_missing_values(column: pd.Series) -> bool:
    """
    This function checks for missing values in the Pandas Series.
    @return: True if missing values exist in column
    """
    return column.hasnans


def count_missing_values(column: pd.Series) -> int:
    """
    This function counts the number of missing values in the Pandas Series.
    @return: sum of boolean 1s if missing value exists.
    """
    return column.isnull().sum()


def number_of_rows(df: pd.Series) -> int:
    """
    This function returns the number of rows in the dataframe.
    @return: numerical value, denoting length of Pandas Series.
    """
    return len(df.index)


def count_negative_values(column: pd.Series) -> int:
    total_neg = (column <= 0).sum()

    return total_neg
