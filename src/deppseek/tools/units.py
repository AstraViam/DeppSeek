"""Dimensional analysis.

Unit errors are the most common class of bug in engineering code and the least
likely to be caught by a test, because a dimensionally wrong result is still a
number. This tool checks an expression's dimensions symbolically, which catches
a mistake the code itself will happily compute.

Requires `pint`. Without it the tool reports that fact rather than guessing.
"""

from __future__ import annotations

from ..errors import ToolError
from .registry import ToolContext, ToolResult, tool


def _registry():
    try:
        import pint
    except ImportError as exc:
        raise ToolError(
            "Unit checking needs pint, which is not installed. "
            "Install it with: python -m pip install pint"
        ) from exc
    return pint.UnitRegistry()


@tool()
def check_units(ctx: ToolContext, expression: str, expected: str = "") -> ToolResult:
    """Evaluate a physical expression and report its dimensions.

    Write quantities as `value * unit`, for example
    "1000 * kg/m**3 * 9.81 * m/s**2 * 10 * m" for hydrostatic pressure.

    Args:
        expression: Expression in pint syntax.
        expected: Optional expected unit, e.g. "Pa" or "W/(m*K)". When given, the
            result is checked against it and any mismatch is reported.
    """
    registry = _registry()
    try:
        value = registry.parse_expression(expression)
    except Exception as exc:
        raise ToolError(
            f"Could not parse {expression!r}: {exc}. "
            f"Write quantities as `value * unit`, e.g. `2.5 * m/s`."
        ) from exc

    lines = [f"Expression: {expression}", f"Result:     {value:~P}"]
    try:
        base = value.to_base_units()
        lines.append(f"SI base:    {base:~P}")
        lines.append(f"Dimensions: {value.dimensionality}")
    except Exception:  # noqa: BLE001 - pint raises many distinct parse error types
        pass

    if expected:
        try:
            converted = value.to(expected)
        except Exception as exc:  # noqa: BLE001 - DimensionalityError and kin
            return ToolResult(
                content="\n".join(lines)
                + f"\n\nDIMENSION MISMATCH: cannot express this as {expected}.\n{exc}\n"
                f"The expression is dimensionally wrong for what it claims to compute.",
                is_error=True,
                display=f"units: mismatch, not {expected}",
            )
        lines.append(f"As {expected}:  {converted:~P}")
        lines.append("Dimensions are consistent with the expected unit.")

    return ToolResult(content="\n".join(lines), display=f"units: {value.units:~P}")


@tool()
def convert_units(ctx: ToolContext, value: float, from_unit: str, to_unit: str) -> ToolResult:
    """Convert a value between units, refusing dimensionally invalid conversions.

    Args:
        value: Numeric magnitude.
        from_unit: Source unit, e.g. "psi".
        to_unit: Target unit, e.g. "bar".
    """
    registry = _registry()
    try:
        quantity = registry.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
    except Exception as exc:
        raise ToolError(f"Cannot convert {value} {from_unit} to {to_unit}: {exc}") from exc
    return ToolResult(
        content=f"{quantity:~P} = {converted:~P}",
        display=f"{value} {from_unit} -> {converted.magnitude:.6g} {to_unit}",
    )
