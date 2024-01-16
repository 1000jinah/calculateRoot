from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

class Series(BaseModel):
    arrayKey: str
    initialValue: float
    periods: int
    value: float
    minusPeriods: int
    minusValue: float

def find_roots(coefficients):
    tolerance = 1e-10
    max_iterations = 1000
    degree = len(coefficients) - 1

    def poly_function(x):
        return sum(c * x**i for i, c in enumerate(coefficients))

    calculated_roots = []

    for i in range(degree + 1):
        a, b = i - 10, i + 11
        iteration = 0

        while iteration < max_iterations:
            c = (a + b) / 2

            if poly_function(a) * poly_function(c) < 0:
                b = c
            else:
                a = c

            if abs(poly_function(c)) < tolerance:
                calculated_roots.append(round(c, 8))
                break

            iteration += 1

    if not calculated_roots:
        raise HTTPException(status_code=400, detail="Roots not found. Check the input coefficients.")

    return calculated_roots[::-1]

@app.post("/calculate_roots")
async def calculate_roots(series_data: List[Series]):
    calculated_roots = []
    first_roots = []
    calculated_values = []
    calculated_values_yearly = []
    balance = []

    for series in series_data:
        coefficients = [series.initialValue]

        coefficients.extend([series.value] * (series.periods * 12))
        coefficients.extend([-series.minusValue] * series.minusPeriods)

        current_calculated_roots = find_roots(coefficients)
        calculated_roots.extend(current_calculated_roots)

        if current_calculated_roots:
            first_root = current_calculated_roots[0]
            first_roots.append(first_root)

            calculated_value = 1 / first_root - 1
            calculated_value_yearly = (1 + calculated_value) ** 12 - 1
            calculated_values.append(calculated_value)
            calculated_values_yearly.append(calculated_value_yearly)

            if 0 < calculated_value_yearly <= 0.15:
                layer = series.periods * 12 + series.minusPeriods
                current_balance = [coefficients.copy()]

                for _ in range(1, layer + 1):
                    current_balance.append([
                        current_balance[-1][j - 1] * (1.0 + calculated_value) + current_balance[-1][j]
                        for j in range(len(current_balance[-1]))
                    ])

                balance.append(current_balance)
            else:
                balance.append([])

    return {
        "calculated_roots": calculated_roots,
        "first_roots": first_roots,
        "calculated_values": calculated_values,
        "calculated_values_yearly": calculated_values_yearly,
        "balance": balance,
    }
