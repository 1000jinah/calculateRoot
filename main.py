from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = FastAPI()

class CashFlowInput(BaseModel):
    initialPayment: float
    monthlyPayment: float
    MonthlyPaymentPeriod: int
    monthlyAllowance: float
    MonthlyAllowancePeriod: int

def find_roots(p):
    p = np.atleast_1d(p)

    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    non_zero = np.nonzero(np.ravel(p))[0]

    if len(non_zero) == 0:
        return np.array([])

    trailing_zeros = len(p) - non_zero[-1] - 1
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]

    if not issubclass(p.dtype.type, (np.floating, np.complexfloating)):
        p = p.astype(float)

    N = len(p)

    if N > 1:
        roots = np.roots(p)
    else:
        roots = np.array([])

    roots = np.hstack((roots, np.zeros(trailing_zeros, dtype=roots.dtype)))
    return roots

def calc_irr(values):
    res = find_roots(values[::-1])

    roots_result = [{'real': root.real, 'imaginary': root.imag} for root in res]

    mask = (res.imag == 0) & (res.real > 0)

    if not mask.any():
        return np.nan

    res = res[mask].real
    rate = 1 / res - 1
    rate = rate.item(np.argmin(np.abs(rate)))
    return rate

def projection_graph(initialPayment, monthlyPayment, MonthlyPaymentPeriod, monthlyAllowance, MonthlyAllowancePeriod):
    sharpe_ratio = 1.0
    z = 0.05
    layer = 5
    cashFlows = [initialPayment]

    for _ in range(MonthlyPaymentPeriod):
        cashFlows.append(monthlyPayment)

    for _ in range(MonthlyAllowancePeriod):
        cashFlows.append(-monthlyAllowance)

    irr = calc_irr(cashFlows)
    sigma = irr * sharpe_ratio
    irr_yearly = (1.0 + irr) ** (12.0) - 1.0
    print("Internal rate of return:%3.4f" % irr_yearly)
    balance = []

    for _ in range(layer):
        balance.append(cashFlows.copy())

        for j in range(MonthlyPaymentPeriod + MonthlyAllowancePeriod):
            n = -layer + 2 * _ + 1
            balance[_][j + 1] = balance[_][j] * (1.0 + irr + n * z * sigma) + balance[_][j + 1]

    df_balance = pd.DataFrame(data=balance)
    # df_balance.T.plot()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    df_balance.T.plot(ax=plt.gca())  # Using the existing axis to prevent creating a new figure
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Encode the image as base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return {"irr_yearly": irr_yearly, "balance": df_balance.to_dict(orient="records"), "plot": image_base64}

@app.post("/calculate-projection")
async def calculate_projection(cash_flow_input: CashFlowInput):
    result = projection_graph(
        cash_flow_input.initialPayment,
        cash_flow_input.monthlyPayment,
        cash_flow_input.MonthlyPaymentPeriod,
        cash_flow_input.monthlyAllowance,
        cash_flow_input.MonthlyAllowancePeriod,
    )
    return result
