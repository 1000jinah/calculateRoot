from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Allow all origins in this example, but you should restrict this to your actual frontend's origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectionGraphInput(BaseModel):
    initialPayment: float
    monthlyPayment: float
    MonthlyPaymentPeriod: int
    monthlyAllowance: float
    MonthlyAllowancePeriod: int

def find_roots(p):
    # If input is scalar, this makes it an array
    p = np.atleast_1d(p)
 
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    # find non-zero array entries
    non_zero = np.nonzero(np.ravel(p))[0]
   
    # Return an empty array if the polynomial is all zeros
    if len(non_zero) == 0:
        return np.array([])
    
    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (np.floating, np.complexfloating)):
        p = p.astype(float)
  
    N = len(p)
    print(p)
    
    if N > 1:
        # Calculate roots of the polynomial
        roots = np.roots(p)
        print("Roots:", roots)
    else:
        roots = np.array([])

    # tack any zeros onto the back of the array
    roots = np.hstack((roots, np.zeros(trailing_zeros, dtype=roots.dtype)))
   
    return roots

def calc_irr(values):
    res = find_roots(values[::-1])
    
    # Convert complex roots to an array of dictionaries
    roots_result = [{'real': root.real, 'imaginary': root.imag} for root in res]

    # Print roots
    # print("Roots:", roots_result)
    
    mask = (res.imag == 0) & (res.real > 0)

    if not mask.any():
        return np.nan
    
    res = res[mask].real
    rate = 1 / res - 1
    rate = rate.item(np.argmin(np.abs(rate)))
    return rate

@app.post("/projection_graph")
async def projection_graph(input_data: ProjectionGraphInput):
    initialPayment = input_data.initialPayment
    monthlyPayment = input_data.monthlyPayment
    MonthlyPaymentPeriod = input_data.MonthlyPaymentPeriod
    monthlyAllowance = input_data.monthlyAllowance
    MonthlyAllowancePeriod = input_data.MonthlyAllowancePeriod

    sharpe_ratio = 1.0
    z = 0.05
    layer = 5
    cashFlows = []
    cashFlows.append(initialPayment)
    for i in range(MonthlyPaymentPeriod):
        cashFlows.append(monthlyPayment)
    for i in range(MonthlyAllowancePeriod):
        cashFlows.append(-monthlyAllowance)

    # Calculate the IRR
    irr = calc_irr(cashFlows)  # monthly IRR
    sigma = irr * sharpe_ratio
    irr_yearly = (1.0 + irr) ** (12.0) - 1.0
    print("Internal rate of return:%3.4f" % irr_yearly)

    balance = []
    for i in range(layer):
        balance.append(cashFlows.copy())
        for j in range(MonthlyPaymentPeriod + MonthlyAllowancePeriod):
            n = -layer + 2 * i + 1
            balance[i][j + 1] = balance[i][j] * (1.0 + irr + n * z * sigma) + balance[i][j + 1]

    return {"irr_yearly": irr_yearly, "balance": balance}