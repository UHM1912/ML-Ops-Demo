from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib, os, json, httpx, mlflow
from mlflow.entities import ViewType
from main import train_with_mlflow
from steps.clean import Cleaner
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently import ColumnMapping

app = FastAPI(
    title="MLops Unified Management API",
    description="A unified API to control model training, experiment tracking, data drift monitoring, and predictions.",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="."), name="static")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_INPUT = {"Gender": "Male", "Age": 35, "HasDrivingLicense": 1, "RegionID": 28.0, "Switch": 0, "PastAccident": "No", "AnnualPremium": 2000.0}

class InputData(BaseModel):
    Gender: str
    Age: int
    HasDrivingLicense: int
    RegionID: float
    Switch: int
    PastAccident: str
    AnnualPremium: float

model = None

def resolve_path(*candidates):
    return next((p for p in candidates if os.path.exists(p)), None)

def load_latest_model():
    global model
    path = resolve_path("models/model.pkl", "backend/models/model.pkl")
    model = joblib.load(path) if path else None

load_latest_model()

# ── Groq helpers ────────────────────────────────────────────────────────────────

async def _groq_post(client: httpx.AsyncClient, groq_key: str, messages: list, json_mode=False) -> str:
    payload = {"model": GROQ_MODEL, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    resp = await client.post(GROQ_URL, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

async def _llm_summarize(client, groq_key, prompt):
    return await _groq_post(client, groq_key, [{"role": "user", "content": prompt + " Write a very short, simple, 1-sentence response. Do not use markdown and use plain text only."}])

# ── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root():
    path = resolve_path("frontend-dist/index.html", "../frontend/dist/frontend/browser/index.html", "frontend/dist/frontend/browser/index.html")
    if path:
        return HTMLResponse(content=open(path, encoding="utf-8").read())
    return HTMLResponse(content="<h3>No frontend interface found.</h3>", status_code=404)

@app.get("/health")
async def health_check():
    return {"status": "ready", "health_check": "OK", "model_loaded": model is not None,
            "message": "Welcome to the Unified MLOps API."}

@app.post("/predict")
async def predict(input_data: InputData):
    if model is None:
        return {"error": "Model not loaded. Please trigger training first."}
    df = pd.DataFrame([input_data.model_dump()])
    return {"predicted_class": int(model.predict(df)[0])}

@app.post("/train")
async def train_model():
    try:
        results = train_with_mlflow()
        load_latest_model()
        return {"status": "success", "message": "Model retrained and loaded successfully.", "metrics": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/experiments")
async def get_experiments():
    try:
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("Model Training Experiment")
        if not exp:
            return {"status": "success", "runs": [], "message": "No experiments found."}
        runs = client.search_runs(experiment_ids=[exp.experiment_id], run_view_type=ViewType.ACTIVE_ONLY,
                                  max_results=6, order_by=["attribute.start_time DESC"])
        return {
            "status": "success", "experiment_name": exp.name, "experiment_id": exp.experiment_id,
            "latest_runs": [{
                "run_id": r.info.run_id, "run_name": r.info.run_name, "status": r.info.status,
                "start_time": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": r.data.metrics, "params": r.data.params
            } for r in runs]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.api_route("/drift", methods=["GET", "POST"])
async def check_drift():
    if model is None:
        return {"error": "Model is not loaded."}
    train_path = resolve_path("data/train.csv", "backend/data/train.csv")
    prod_path  = resolve_path("data/production.csv", "backend/data/production.csv")
    if not train_path or not prod_path:
        return {"error": "Datasets (train.csv or production.csv) not found."}
    try:
        cleaner = Cleaner()
        ref  = cleaner.clean_data(pd.read_csv(train_path))
        prod = cleaner.clean_data(pd.read_csv(prod_path))
        ref["prediction"]  = model.predict(ref.iloc[:, :-1])
        prod["prediction"] = model.predict(prod.iloc[:, :-1])

        cm = ColumnMapping(target="Result", prediction="prediction",
                           numerical_features=["Age", "AnnualPremium", "HasDrivingLicense", "RegionID", "Switch"],
                           categorical_features=["Gender", "PastAccident"])
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
        report.run(reference_data=ref, current_data=prod, column_mapping=cm)
        report.save_html("production_drift.html")

        drift = next((m["result"] for m in report.as_dict()["metrics"] if m["metric"] == "DatasetDriftMetric"), {})
        return {"status": "success", "report_file": "production_drift.html", "message": "Report generated.",
                "summary": {"drift_detected": drift.get("dataset_drift", False),
                            "number_of_drifted_features": drift.get("number_of_drifted_columns", 0),
                            "share_of_drifted_features": drift.get("share_of_drifted_columns", 0.0),
                            "total_features": drift.get("number_of_columns", 0)}}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ── Chat ─────────────────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are an MLOps routing assistant. Classify intent into one of: TRAIN, METRICS, DRIFT, PREDICT, CHAT.
For the response field, keep it extremely brief, simple, and clean (maximum 1 short sentence). No long paragraphs.
Respond in raw JSON: {"action": "...", "reasoning": "...", "response": "..."}
No markdown formatting."""

LOCAL_KEYWORDS = {
    "TRAIN":   ["train", "retrain", "fit", "run training", "re-train"],
    "METRICS": ["metric", "experiment", "run", "mlflow", "performance"],
    "DRIFT":   ["drift", "evidently", "distribution", "production", "quality"],
    "PREDICT": ["predict", "customer", "buy", "cross-sell", "insurance", "prediction"],
}

def _local_classify(msg: str) -> str:
    msg = msg.lower()
    return next((action for action, kws in LOCAL_KEYWORDS.items() if any(w in msg for w in kws)), "CHAT")

async def _execute_action(action: str, user_message: str, client=None, groq_key=None):
    """Run the pipeline action and return (agent_reply, raw_result, extra)."""
    if action == "TRAIN":
        res = await train_model()
        reply = (await _llm_summarize(client, groq_key, f"Retraining complete: {res}.")
                 if groq_key else
                 "Model retraining completed successfully.")
        return reply, res, {}

    if action == "METRICS":
        res = await get_experiments()
        if groq_key:
            reply = await _llm_summarize(client, groq_key, f"MLflow runs summary: {res}.")
        else:
            runs = res.get("latest_runs", [])
            lines = [f"- {r['run_name']} ({r['run_id'][:8]}): Acc={r['metrics'].get('accuracy',0):.4f}" for r in runs]
            reply = ("Latest runs:\n" + "\n".join(lines)) if lines else "No runs found."
        return reply, res, {}

    if action == "DRIFT":
        res = await check_drift()
        if groq_key:
            reply = await _llm_summarize(client, groq_key, f"Drift analysis complete: {res}.")
        else:
            s = res.get("summary", {})
            status = "detected" if s.get("drift_detected") else "clear"
            reply = f"Drift analysis completed. Status: {status}."
        return reply, res, {}

    if action == "PREDICT":
        if groq_key:
            raw = await _groq_post(client, groq_key,
                [{"role": "user", "content": f"Extract customer fields from: '{user_message}'. Return JSON with keys: Gender, Age, HasDrivingLicense, RegionID, Switch, PastAccident, AnnualPremium."}],
                json_mode=True)
            inputs = {**DEFAULT_INPUT, **{k: v for k, v in json.loads(raw).items() if v not in (None, "")}}
        else:
            import re
            msg = user_message.lower()
            age_m = re.search(r'\bage\b\s*(\d+)|\b(\d+)\s*years?\b|\b(?!premium\b)(\d{2})\b', msg)
            pre_m = re.search(r'\bpremium\b\s*(\d+)|\b(\d{4,5})\b', msg)
            inputs = {**DEFAULT_INPUT,
                      "Gender": "Female" if "female" in msg else "Male",
                      "PastAccident": "Yes" if "accident" in msg and "no" not in msg else "No",
                      "Age": int(next(g for g in age_m.groups() if g)) if age_m else 35,
                      "AnnualPremium": float(next(g for g in pre_m.groups() if g)) if pre_m else 2500.0}

        res = await predict(InputData(**inputs))
        if groq_key:
            reply = await _llm_summarize(client, groq_key, f"Customer: {inputs}, prediction: {res}. Explain result.")
        else:
            label = "likely" if res.get("predicted_class") == 1 else "unlikely"
            reply = f"Prediction result: The customer is {label} to purchase insurance."
        return reply, res, {"inputs": inputs}

    # CHAT
    return "Hello! I can retrain models, check data drift, or query metrics.", {}, {}

@app.post("/chat")
async def chat_endpoint(payload: dict):
    user_message = payload.get("message", "")
    if not user_message:
        return {"error": "Empty message."}

    groq_key = os.environ.get("GROQ_API_KEY")

    try:
        if not groq_key:
            action = _local_classify(user_message)
            reply, res, extra = await _execute_action(action, user_message)
            return {"response": reply, "action": action, "data": res, **extra}

        async with httpx.AsyncClient() as client:
            intent_raw = await _groq_post(client, groq_key,
                [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
                json_mode=True)
            intent = json.loads(intent_raw)
            action = intent.get("action", "CHAT")

            if action == "CHAT":
                return {"response": intent.get("response", "How can I help?"), "action": "CHAT"}

            reply, res, extra = await _execute_action(action, user_message, client, groq_key)
            return {"response": reply, "action": action, "data": res, **extra}

    except Exception as e:
        return {"error": str(e)}

# ── Angular static bundle ────────────────────────────────────────────────────────
angular_dir = resolve_path("frontend-dist", "../frontend/dist/frontend/browser", "frontend/dist/frontend/browser")
if angular_dir:
    app.mount("/", StaticFiles(directory=angular_dir), name="angular")