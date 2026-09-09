"""
Interactive Web Application for Customer Churn Prediction.
Built with FastAPI and modern responsive UI.
Run with: python app.py
"""

import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

from src.predict import ChurnPredictor


app = FastAPI(title="Customer Churn Prediction AI Dashboard")

# Mount reports directory for images if exists
os.makedirs("reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        if os.path.exists("models/best_model.joblib") and os.path.exists("models/preprocessor.joblib"):
            predictor = ChurnPredictor()
    return predictor


class CustomerInput(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 4
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 85.0
    TotalCharges: float = 340.0


@app.get("/api/metrics")
async def get_metrics():
    if os.path.exists("models/metrics_summary.json"):
        with open("models/metrics_summary.json") as f:
            return json.load(f)
    return {"error": "Models have not been trained yet. Please run main.py first."}


@app.post("/api/predict")
async def predict_churn(customer: CustomerInput):
    pred = get_predictor()
    if pred is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Model not trained yet. Run `python main.py` first to train models."},
        )
    result = pred.predict_customer(customer.model_dump())
    return result


@app.get("/", response_class=HTMLResponse)
async def home_page():
    metrics_json = {}
    if os.path.exists("models/metrics_summary.json"):
        with open("models/metrics_summary.json") as f:
            metrics_json = json.load(f)

    best_model_name = metrics_json.get("best_model", "XGBoost")
    models_data = metrics_json.get("models", {})

    # Generate table rows
    table_rows = ""
    for name, data in models_data.items():
        is_best = "★ BEST" if name == best_model_name else ""
        badge_style = "background: #10b981; color: white; padding: 2px 8px; border-radius: 9999px; font-size: 11px; margin-left: 8px;" if is_best else ""
        table_rows += f"""
        <tr>
            <td style="font-weight: 600;">{name} <span style="{badge_style}">{is_best}</span></td>
            <td>{data['accuracy']*100:.2f}%</td>
            <td>{data['precision']:.4f}</td>
            <td>{data['recall']:.4f}</td>
            <td style="font-weight: 700; color: #3b82f6;">{data['f1_score']:.4f}</td>
            <td>{data['roc_auc']:.4f}</td>
            <td>TP:{data['confusion_matrix']['tp']} | FP:{data['confusion_matrix']['fp']}<br>FN:{data['confusion_matrix']['fn']} | TN:{data['confusion_matrix']['tn']}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customer Churn Prediction AI Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0f172a;
                --card-bg: #1e293b;
                --card-border: #334155;
                --primary: #3b82f6;
                --primary-hover: #2563eb;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --danger: #ef4444;
                --warning: #f59e0b;
                --success: #10b981;
            }}
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Plus Jakarta Sans', sans-serif;
            }}
            body {{
                background-color: var(--bg);
                color: var(--text);
                padding: 24px;
                line-height: 1.5;
            }}
            .container {{
                max-width: 1280px;
                margin: 0 auto;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 28px;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--card-border);
            }}
            .header-title h1 {{
                font-size: 26px;
                font-weight: 800;
                background: linear-gradient(135deg, #60a5fa, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .header-title p {{
                color: var(--text-muted);
                font-size: 14px;
                margin-top: 4px;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 14px;
                border-radius: 9999px;
                font-size: 12px;
                font-weight: 600;
                background: rgba(59, 130, 246, 0.15);
                color: #60a5fa;
                border: 1px solid rgba(59, 130, 246, 0.3);
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 24px;
            }}
            @media (max-width: 900px) {{
                .grid {{ grid-template-columns: 1fr; }}
            }}
            .card {{
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            }}
            .card-title {{
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
                text-align: left;
            }}
            th {{
                color: var(--text-muted);
                font-weight: 600;
                padding: 10px 8px;
                border-bottom: 1px solid var(--card-border);
            }}
            td {{
                padding: 12px 8px;
                border-bottom: 1px solid rgba(51, 65, 85, 0.5);
            }}
            .form-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
            }}
            .form-group {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}
            label {{
                font-size: 12px;
                font-weight: 600;
                color: var(--text-muted);
            }}
            input, select {{
                background: #0f172a;
                border: 1px solid var(--card-border);
                color: var(--text);
                padding: 10px 12px;
                border-radius: 8px;
                font-size: 13px;
                outline: none;
                transition: border 0.2s;
            }}
            input:focus, select:focus {{
                border-color: var(--primary);
            }}
            .btn {{
                background: var(--primary);
                color: white;
                border: none;
                padding: 14px 20px;
                border-radius: 10px;
                font-weight: 700;
                cursor: pointer;
                width: 100%;
                margin-top: 18px;
                font-size: 15px;
                transition: all 0.2s;
            }}
            .btn:hover {{
                background: var(--primary-hover);
                transform: translateY(-1px);
            }}
            .result-card {{
                display: none;
                background: rgba(15, 23, 42, 0.8);
                border-radius: 12px;
                padding: 18px;
                margin-top: 20px;
                border: 1px solid var(--card-border);
            }}
            .result-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }}
            .risk-pill {{
                padding: 4px 12px;
                border-radius: 9999px;
                font-weight: 700;
                font-size: 12px;
            }}
            .progress-bar-bg {{
                background: #334155;
                border-radius: 9999px;
                height: 12px;
                overflow: hidden;
                margin: 10px 0 16px 0;
            }}
            .progress-bar-fill {{
                height: 100%;
                width: 0%;
                border-radius: 9999px;
                transition: width 0.6s ease;
            }}
            .recommendation-list {{
                list-style: none;
                font-size: 13px;
            }}
            .recommendation-list li {{
                padding: 6px 0;
                color: #cbd5e1;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .recommendation-list li::before {{
                content: "💡";
            }}
            .img-container {{
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--card-border);
                margin-top: 12px;
            }}
            .img-container img {{
                width: 100%;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="header-title">
                    <h1>Customer Churn Prediction Dashboard</h1>
                    <p>Machine Learning Multi-Model Classification & Retention Engine</p>
                </div>
                <div>
                    <span class="badge">Production Pipeline Active</span>
                </div>
            </header>

            <!-- Models Performance Section -->
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-title">
                    <span>📊</span> Model Benchmark Comparison (Test Set)
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>ROC-AUC</th>
                                <th>Confusion Matrix Breakdown</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows or '<tr><td colspan="7">No metrics found. Run main.py to train models.</td></tr>'}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Main Interactive Section -->
            <div class="grid">
                <!-- Prediction Form -->
                <div class="card">
                    <div class="card-title">
                        <span>⚡</span> Predict Churn Risk for Customer
                    </div>
                    <form id="churnForm">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Contract Type</label>
                                <select id="Contract">
                                    <option value="Month-to-month" selected>Month-to-month</option>
                                    <option value="One year">One year</option>
                                    <option value="Two year">Two year</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Tenure (Months: 1 - 72)</label>
                                <input type="number" id="tenure" value="4" min="1" max="72">
                            </div>
                            <div class="form-group">
                                <label>Monthly Charges ($)</label>
                                <input type="number" step="0.5" id="MonthlyCharges" value="85.50">
                            </div>
                            <div class="form-group">
                                <label>Total Charges ($)</label>
                                <input type="number" step="1.0" id="TotalCharges" value="342.00">
                            </div>
                            <div class="form-group">
                                <label>Internet Service</label>
                                <select id="InternetService">
                                    <option value="Fiber optic" selected>Fiber optic</option>
                                    <option value="DSL">DSL</option>
                                    <option value="No">No Internet</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Tech Support</label>
                                <select id="TechSupport">
                                    <option value="No" selected>No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Online Security</label>
                                <select id="OnlineSecurity">
                                    <option value="No" selected>No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Payment Method</label>
                                <select id="PaymentMethod">
                                    <option value="Electronic check" selected>Electronic check</option>
                                    <option value="Mailed check">Mailed check</option>
                                    <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                                    <option value="Credit card (automatic)">Credit card (automatic)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Paperless Billing</label>
                                <select id="PaperlessBilling">
                                    <option value="Yes" selected>Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Senior Citizen</label>
                                <select id="SeniorCitizen">
                                    <option value="0" selected>No (0)</option>
                                    <option value="1">Yes (1)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Partner</label>
                                <select id="Partner">
                                    <option value="No" selected>No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Dependents</label>
                                <select id="Dependents">
                                    <option value="No" selected>No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                        </div>

                        <button type="submit" class="btn" id="predictBtn">Evaluate Churn Risk</button>
                    </form>

                    <!-- Prediction Result View -->
                    <div class="result-card" id="resultCard">
                        <div class="result-header">
                            <div>
                                <span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Prediction</span>
                                <h3 id="churnResultText" style="font-size: 20px; font-weight: 800;">-</h3>
                            </div>
                            <span class="risk-pill" id="riskPill">-</span>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600;">
                                <span>Churn Probability</span>
                                <span id="probPercent">0%</span>
                            </div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" id="probFill"></div>
                            </div>
                        </div>
                        <div style="margin-top: 14px;">
                            <span style="font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Recommended Retention Strategies:</span>
                            <ul class="recommendation-list" id="recommendationsList" style="margin-top: 8px;"></ul>
                        </div>
                    </div>
                </div>

                <!-- Visual Charts View -->
                <div class="card">
                    <div class="card-title">
                        <span>📈</span> Confusion Matrices & Model ROC
                    </div>
                    <div class="img-container">
                        <img src="/reports/confusion_matrices.png" onerror="this.src='https://placehold.co/600x400/1e293b/94a3b8?text=Run+main.py+to+generate+reports'" alt="Confusion Matrices">
                    </div>
                    <div class="img-container" style="margin-top: 16px;">
                        <img src="/reports/roc_curves.png" onerror="this.src='https://placehold.co/600x400/1e293b/94a3b8?text=Run+main.py+to+generate+reports'" alt="ROC Curves">
                    </div>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('churnForm');
            const resultCard = document.getElementById('resultCard');
            const churnResultText = document.getElementById('churnResultText');
            const riskPill = document.getElementById('riskPill');
            const probPercent = document.getElementById('probPercent');
            const probFill = document.getElementById('probFill');
            const recommendationsList = document.getElementById('recommendationsList');
            const predictBtn = document.getElementById('predictBtn');

            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                predictBtn.disabled = true;
                predictBtn.innerText = "Analyzing Risk Profile...";

                const payload = {{
                    gender: "Female",
                    SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                    Partner: document.getElementById('Partner').value,
                    Dependents: document.getElementById('Dependents').value,
                    tenure: parseInt(document.getElementById('tenure').value),
                    PhoneService: "Yes",
                    MultipleLines: "No",
                    InternetService: document.getElementById('InternetService').value,
                    OnlineSecurity: document.getElementById('OnlineSecurity').value,
                    OnlineBackup: "No",
                    DeviceProtection: "No",
                    TechSupport: document.getElementById('TechSupport').value,
                    StreamingTV: "Yes",
                    StreamingMovies: "Yes",
                    Contract: document.getElementById('Contract').value,
                    PaperlessBilling: document.getElementById('PaperlessBilling').value,
                    PaymentMethod: document.getElementById('PaymentMethod').value,
                    MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                    TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
                }};

                try {{
                    const response = await fetch('/api/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }});
                    const data = await response.json();
                    
                    if (data.error) {{
                        alert(data.error);
                        return;
                    }}

                    resultCard.style.display = 'block';
                    churnResultText.innerText = data.churn_prediction;
                    riskPill.innerText = data.risk_tier;
                    riskPill.style.background = data.risk_color + '22';
                    riskPill.style.color = data.risk_color;
                    riskPill.style.border = '1px solid ' + data.risk_color;

                    probPercent.innerText = data.churn_probability_percentage;
                    probFill.style.width = data.churn_probability_percentage;
                    probFill.style.background = data.risk_color;

                    recommendationsList.innerHTML = '';
                    data.recommendations.forEach(rec => {{
                        const li = document.createElement('li');
                        li.innerText = rec;
                        recommendationsList.appendChild(li);
                    }});

                }} catch (err) {{
                    console.error(err);
                    alert("Error predicting churn: " + err.message);
                }} finally {{
                    predictBtn.disabled = false;
                    predictBtn.innerText = "Evaluate Churn Risk";
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
