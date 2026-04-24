from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.service import PortfolioService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    split_ratio: float

service = PortfolioService()

@app.post("/api/analyze")
async def analyze_portfolio(req: AnalysisRequest):
    try:
        return service.run_analysis(
            req.tickers,
            req.start_date,
            req.end_date,
            req.split_ratio
        )
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)