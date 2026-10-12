"""
Uncertainty-Guided Sensor Placement — Streamlit app

Run:
    pip install -r requirements.txt
    streamlit run app.py

Needs soil_moisture_clean.csv (saved by the last cell of the notebook) in the same folder.
The first run also downloads a coarse ERA5-Land grid for the map and saves it as era5_grid.csv,
so later runs start instantly.
"""

import json
import os
import re
import time

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from streamlit_folium import st_folium

DATA_FILE = "soil_moisture_clean.csv"
GRID_FILE = "era5_grid.csv"
BEGIN_DATE = "2023-06-01"
END_DATE = "2023-06-30"
OM_URL = "https://archive-api.open-meteo.com/v1/archive"
JITTER = 1e-8

# Section 13 of the notebook, averaged over Apr/Jun/Aug/Oct 2023 (MAE, % soil moisture)
EXPERIMENT = {
    5:  {"random, no prior": 8.29, "random + prior": 8.65, "MI + prior": 7.66, "calib-MI + prior": 7.21},
    10: {"random, no prior": 7.64, "random + prior": 7.32, "MI + prior": 7.19, "calib-MI + prior": 7.01},
    20: {"random, no prior": 7.07, "random + prior": 6.85, "MI + prior": 6.77, "calib-MI + prior": 6.81},
    30: {"random, no prior": 6.84, "random + prior": 6.68, "MI + prior": 6.68, "calib-MI + prior": 6.67},
}
EXPERIMENT_WINS = "At 5 sensors, calibration-aware MI + prior beat random placement without the prior in 167 of 200 paired runs (p < 0.0001)."

CRITERIA = ["Calibration-aware MI (best)", "Mutual information", "Entropy (least confident)"]


# ----------------------------------------------------------------------------- data

def fetch_era5(lats, lons, batch=25, pause=6):
    """Monthly-mean ERA5-Land soil moisture (%) at each point. NaN over sea / missing."""
    out = []
    for start in range(0, len(lats), batch):
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in lats[start:start + batch]),
            "longitude": ",".join(f"{v:.4f}" for v in lons[start:start + batch]),
            "start_date": BEGIN_DATE,
            "end_date": END_DATE,
            "hourly": "soil_moisture_0_to_7cm",
            "models": "era5_land",
        }
        for attempt in range(5):
            r = requests.get(OM_URL, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(60)
                continue
            if r.status_code == 400 and "models" in params:
                params.pop("models")
                continue
            break
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict):
            js = [js]
        for loc in js:
            vals = [v for v in loc["hourly"]["soil_moisture_0_to_7cm"] if v is not None]
            out.append(100.0 * sum(vals) / len(vals) if vals else np.nan)
        if start + batch < len(lats):
            time.sleep(pause)
    return np.array(out)


@st.cache_data(show_spinner="Loading stations...")
def load_stations():
    df = pd.read_csv(DATA_FILE)
    if "era5" not in df.columns or df["era5"].isna().all():
        df["era5"] = fetch_era5(list(df.latitude), list(df.longitude))
    df = df[df.era5.notna()].reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Downloading the coarse ERA5 grid for the map (first run only, ~1 min)...")
def load_grid(lon_min, lon_max, lat_min, lat_max):
    if os.path.exists(GRID_FILE):
        return pd.read_csv(GRID_FILE)
    lons = np.linspace(lon_min, lon_max, 26)
    lats = np.linspace(lat_min, lat_max, 14)
    rows = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            rows.append({"latitude": lats[i], "longitude": lons[j]})
    grid = pd.DataFrame(rows)
    grid["era5"] = fetch_era5(list(grid.latitude), list(grid.longitude))
    grid = grid[grid.era5.notna()].reset_index(drop=True)     # sea cells drop out -> land only
    grid.to_csv(GRID_FILE, index=False)
    return grid


# ----------------------------------------------------------------------------- model

def features(df):
    return df[["longitude", "latitude"]].values


def design(prior):
    """Rows [1, ERA5/10] — inputs of the calibration line station = a + b*ERA5."""
    return np.column_stack([np.ones(len(prior)), np.asarray(prior) / 10.0])


def linear_fit(xv, yv):
    n = len(xv)
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx = mx + xv[i]
        my = my + yv[i]
    mx = mx / n
    my = my / n
    sxy = 0.0
    sxx = 0.0
    for i in range(n):
        sxy = sxy + (xv[i] - mx) * (yv[i] - my)
        sxx = sxx + (xv[i] - mx) * (xv[i] - mx)
    b = sxy / sxx if sxx > 0 else 1.0
    a = my - b * mx
    return a, b


@st.cache_resource(show_spinner="Learning the GP kernel...")
def learn_kernel(X, resid):
    """Hyperparameters learned once from the full historical network (like FIXED_KERNEL)."""
    k = ConstantKernel(1.0) * Matern(length_scale=5.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=3, normalize_y=False)
    gp.fit(X, (resid - resid.mean()) / resid.std())
    return gp.kernel_


def fit_model(kernel, X, y, prior, s):
    """Calibrated prior + GP on the residuals (the notebook's best mode).
    s = residual spread of the full network, so uncertainty is on the same scale the kernel was learned on."""
    a, b = linear_fit(prior, y)
    resid = y - (a + b * prior)
    m = resid.mean()
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gp.fit(X, (resid - m) / s)
    return {"a": a, "b": b, "m": m, "s": s, "gp": gp, "X": X, "F": design(prior), "kernel": kernel}


def uk_var(kernel, Xq, Xs, Fq, Fs, latent=False):
    """GP variance + uncertainty of the calibration line (universal kriging).
    latent=True leaves out sensor noise: 'how unsure are we about the TRUE soil moisture here'.
    That is what the map shows, because noise can't be removed by adding sensors."""
    Kss = kernel(Xs) + JITTER * np.eye(len(Xs))
    Kqs = kernel(Xq, Xs)
    A = np.linalg.solve(Kss, Kqs.T)
    prior_var = kernel.k1.diag(Xq) if latent else kernel.diag(Xq)
    gp_var = prior_var - np.sum(Kqs * A.T, axis=1)
    B = np.linalg.solve(Kss, Fs)
    M = Fs.T @ B + 1e-6 * np.eye(2)
    R = Fq - A.T @ Fs
    trend = np.sum((R @ np.linalg.inv(M)) * R, axis=1)
    return np.maximum(gp_var + trend, 1e-12)


def post_var(kernel, Xq, Xs):
    Kss = kernel(Xs) + JITTER * np.eye(len(Xs))
    Kqs = kernel(Xq, Xs)
    v = np.linalg.solve(Kss, Kqs.T)
    return np.maximum(kernel.diag(Xq) - np.sum(Kqs * v.T, axis=1), 1e-12)


def var_rest(kernel, Xc):
    K = kernel(Xc) + JITTER * np.eye(len(Xc))
    return np.maximum(1.0 / np.diag(np.linalg.inv(K)), 1e-12)


def predict(model, Xq, prior_q, latent=False):
    base = model["a"] + model["b"] * np.asarray(prior_q)
    mean = base + model["gp"].predict(Xq) * model["s"] + model["m"]
    std = np.sqrt(uk_var(model["kernel"], Xq, model["X"], design(prior_q), model["F"], latent)) * model["s"]
    return mean, std


def greedy_pick(kernel, Xk, Fk, Xc, Fc, k, criterion):
    """Greedy placement. Xk/Fk = locations that already have a sensor, Xc/Fc = candidate locations.
    Returns the chosen candidate positions (indices into Xc), best first.
    Note: uncertainty only depends on WHERE sensors are, not on their readings, so this works
    for brand-new sites that have no reading yet."""
    Xk = np.array(Xk)
    Fk = np.array(Fk)
    cand = list(range(len(Xc)))
    chosen = []
    for step in range(k):
        if len(cand) == 0:
            break
        Xcc = Xc[cand]
        Fcc = Fc[cand]
        if criterion == CRITERIA[0]:
            score = uk_var(kernel, Xcc, Xk, Fcc, Fk) / var_rest(kernel, Xcc)
        elif criterion == CRITERIA[1]:
            score = post_var(kernel, Xcc, Xk) / var_rest(kernel, Xcc)
        else:
            score = post_var(kernel, Xcc, Xk)
        best = int(np.argmax(score))
        pick = cand[best]
        chosen.append(pick)
        Xk = np.vstack([Xk, Xc[pick:pick + 1]])
        Fk = np.vstack([Fk, Fc[pick:pick + 1]])
        cand.pop(best)
    return chosen


def check_readings(kernel, X, y, prior, s):
    """Leave-one-out check: for every station, what would its neighbours + the calibrated prior
    have predicted if it didn't exist? A big gap (|z| > 2) = reading worth checking.
    Uses the standard GP shortcut: LOO error = alpha_i / (K^-1)_ii, LOO variance = 1 / (K^-1)_ii."""
    a, b = linear_fit(prior, y)
    resid = y - (a + b * prior)
    r = (resid - resid.mean()) / s
    Kinv = np.linalg.inv(kernel(X) + JITTER * np.eye(len(X)))
    alpha = Kinv @ r
    expected = np.zeros(len(y))
    z = np.zeros(len(y))
    for i in range(len(y)):
        loo_err = alpha[i] / Kinv[i, i]              # (actual - predicted), standardised
        loo_std = np.sqrt(1.0 / Kinv[i, i])
        z[i] = loo_err / loo_std
        expected[i] = y[i] - loo_err * s
    return expected, z


def nearest_station(df, lat, lon):
    best_i = 0
    best_d = 1e18
    for i in range(len(df)):
        d = (df.latitude.iloc[i] - lat) ** 2 + (df.longitude.iloc[i] - lon) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return df.iloc[best_i]


def km_between(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 110.57
    return float(np.sqrt(dx * dx + dy * dy))


def describe_site(df, lat, lon):
    near = nearest_station(df, lat, lon)
    d = km_between(lat, lon, near.latitude, near.longitude)
    return f"lat {lat:.2f}, lon {lon:.2f} — {d:.0f} km from the nearest sensor ({near['name']}, {near['stateCode']})"


# ----------------------------------------------------------------------------- chatbot tools
# The LLM only CHOOSES one of these. The numbers in every answer come from this code.

def tool_recommend(ctx, k=3):
    k = max(1, min(int(k), 10))
    picks = ctx["recommend_fn"](k)
    if ctx["mode"] == "real":
        lines = [f"Using **{ctx['criterion']}**, the {len(picks)} best NEW sites for a sensor are:"]
    else:
        lines = [f"Using **{ctx['criterion']}**, the next {len(picks)} sensor(s) should go at:"]
    for r in range(len(picks)):
        lines.append(f"{r + 1}. {picks[r]}")
    return "\n".join(lines)


def tool_uncertain(ctx, k=3):
    k = max(1, min(int(k), 10))
    order = np.argsort(-ctx["grid_std"])
    lines = [f"The {k} areas the model is least sure about (± = one standard deviation):"]
    for r in range(k):
        i = order[r]
        lat = ctx["grid"].latitude.iloc[i]
        lon = ctx["grid"].longitude.iloc[i]
        lines.append(f"{r + 1}. {describe_site(ctx['df_known'], lat, lon)} — "
                     f"{ctx['grid_mean'][i]:.1f}% ± {ctx['grid_std'][i]:.1f}%")
    return "\n".join(lines)


def tool_suspicious(ctx, k=5):
    if ctx["mode"] != "real":
        return "Switch to **Real network** mode to check the readings of all stations."
    sus = ctx["suspicious"]
    if len(sus) == 0:
        return "No station disagrees strongly (|z| > 2) with its neighbours and the coarse estimate."
    k = max(1, min(int(k), len(sus)))
    lines = [f"{len(sus)} station(s) disagree strongly with what their neighbours + the coarse estimate predict. "
             f"Top {k}:"]
    for r in range(k):
        row = sus.iloc[r]
        lines.append(f"{r + 1}. {row['name']} ({row['stateCode']}) reads {row['moisture']:.1f}%, "
                     f"expected ~{row['expected']:.1f}% (z = {row['z']:+.1f})")
    lines.append("\nThis means *worth checking*, not *wrong*: it can be a faulty sensor, or genuinely different "
                 "local soil (clay, irrigation, a wetland).")
    return "\n".join(lines)


def tool_predict(ctx, lat, lon):
    lat = float(lat)
    lon = float(lon)
    if not (24 <= lat <= 50 and -125 <= lon <= -66):
        return "That point is outside the continental US, where this model has no data."
    try:
        prior = fetch_era5([lat], [lon], pause=0)
    except Exception:
        return "I couldn't fetch the coarse ERA5 estimate for that point right now, so I can't predict there."
    if np.isnan(prior[0]):
        return "ERA5-Land has no soil value there (probably water), so I can't predict at that point."
    mean, std = predict(ctx["model"], np.array([[lon, lat]]), prior)
    return (f"Predicted soil moisture at lat {lat:.2f}, lon {lon:.2f}: **{mean[0]:.1f}% ± {std[0]:.1f}%** "
            f"(coarse ERA5 estimate there was {prior[0]:.1f}%; the ground sensors corrected it).")


def tool_compare(ctx):
    lines = ["Leakage-free experiment, averaged over Apr/Jun/Aug/Oct 2023 (lower error is better):", "",
             "| sensors | random, no prior | random + prior | MI + prior | calib-MI + prior |",
             "|---|---|---|---|---|"]
    for n in EXPERIMENT:
        e = EXPERIMENT[n]
        lines.append(f"| {n} | {e['random, no prior']:.2f} | {e['random + prior']:.2f} | "
                     f"{e['MI + prior']:.2f} | {e['calib-MI + prior']:.2f} |")
    lines.append("")
    lines.append(EXPERIMENT_WINS)
    return "\n".join(lines)


def tool_summary(ctx):
    return ctx["summary"]


TOOLS = {
    "recommend_next": ("Where to place the next / new sensor(s). args: k (int, default 3)", tool_recommend),
    "most_uncertain_areas": ("Which areas the model is least confident about / coverage gaps. args: k (int, default 3)", tool_uncertain),
    "suspicious_readings": ("Which stations' readings look off / inaccurate / faulty. args: k (int, default 5)", tool_suspicious),
    "predict_at": ("Predicted soil moisture at a location. args: lat (float), lon (float)", tool_predict),
    "compare_strategies": ("How the placement methods compare / proof the method works. no args", tool_compare),
    "network_summary": ("How many sensors, overall uncertainty, overall status. no args", tool_summary),
}


def route_with_llm(question, api_key, model_name):
    tool_list = "\n".join(f"- {name}: {TOOLS[name][0]}" for name in TOOLS)
    prompt = (
        "You route questions for a soil-moisture sensor-placement app. Pick exactly ONE tool.\n"
        f"Tools:\n{tool_list}\n- none: the question is not about this sensor network\n\n"
        "Reply with JSON only, like {\"tool\": \"recommend_next\", \"args\": {\"k\": 2}}.\n"
        "For place names, convert to approximate lat/lon yourself.\n\n"
        f"Question: {question}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
    r = requests.post(url, params={"key": api_key}, json=body, timeout=30)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    choice = json.loads(text)
    return choice.get("tool", "none"), choice.get("args", {}) or {}


def route_with_keywords(question):
    q = question.lower()
    nums = re.findall(r"-?\d+(?:\.\d+)?", q)
    k = None
    m = re.search(r"\b(\d+)\b", q)
    if m and int(m.group(1)) <= 10:
        k = int(m.group(1))
    if len(nums) >= 2 and ("lat" in q or "predict" in q or "moisture at" in q or "," in q):
        return "predict_at", {"lat": nums[0], "lon": nums[1]}
    if any(w in q for w in ["suspicious", "faulty", "wrong", "inaccurate", "accurate", "off", "outlier", "check"]):
        return "suspicious_readings", ({"k": k} if k else {})
    if any(w in q for w in ["compare", "random", "better", "work", "proof", "baseline"]):
        return "compare_strategies", {}
    if any(w in q for w in ["unsure", "uncertain", "confident", "don't know", "weak", "gap", "lack", "missing"]):
        return "most_uncertain_areas", ({"k": k} if k else {})
    if any(w in q for w in ["where", "next", "place", "put", "deploy", "add", "new"]):
        return "recommend_next", ({"k": k} if k else {})
    if any(w in q for w in ["how many", "summary", "status", "error", "overall"]):
        return "network_summary", {}
    return "none", {}


def answer(question, ctx, api_key, model_name):
    tool, args = None, {}
    note = ""
    if api_key:
        try:
            tool, args = route_with_llm(question, api_key, model_name)
        except Exception as e:
            note = f"\n\n_(LLM unavailable: {type(e).__name__}. Used keyword matching instead.)_"
    if tool is None:
        tool, args = route_with_keywords(question)
    if tool not in TOOLS:
        return ("I can only answer questions about this sensor network — try *“Where should new sensors go?”*, "
                "*“Where are we missing sensors?”*, *“Which readings look off?”*, *“Moisture at 35.2, -101.8”* "
                "or *“Does this beat random placement?”*" + note)
    try:
        return TOOLS[tool][1](ctx, **args) + note
    except Exception as e:
        return f"That calculation failed ({type(e).__name__}: {e})." + note


# ----------------------------------------------------------------------------- page

st.set_page_config(page_title="Sensor Placement", page_icon="🛰️", layout="wide")
st.title("Uncertainty-Guided Sensor Placement")
st.caption("Soil moisture, June 2023 · SCAN ground stations + ERA5-Land coarse estimate · calibrated prior + Gaussian Process")

if not os.path.exists(DATA_FILE):
    st.error(f"`{DATA_FILE}` not found. Run the notebook, download the CSV from its last cell and put it next to app.py.")
    st.stop()

df = load_stations()
X_all = features(df)
y_all = df["moisture"].values
p_all = df["era5"].values
F_all = design(p_all)

a0, b0 = linear_fit(p_all, y_all)
resid_all = y_all - (a0 + b0 * p_all)
kernel = learn_kernel(X_all, resid_all)
S_HIST = float(resid_all.std())

grid = load_grid(float(df.longitude.min()), float(df.longitude.max()),
                 float(df.latitude.min()), float(df.latitude.max()))
X_grid = features(grid)
F_grid = design(grid["era5"].values)

# ---- sidebar
with st.sidebar:
    mode_label = st.radio("Mode", ["Real network", "Proof (simulation)"], key="mode",
                          help="Real network uses every real sensor. Proof hides most of them to show the method works.")
    mode = "real" if mode_label == "Real network" else "proof"

    st.header("Placement")
    criterion = st.selectbox("Criterion", CRITERIA)
    k_rec = st.slider("How many to recommend", 1, 10, 3)

    if mode == "proof":
        st.header("Simulation")
        n_start = st.slider("Pretend only this many sensors exist", 3, 60, 8)
        seed = st.number_input("Random seed", 0, 999, 0)
        if st.button("Reset simulation", key="reset") or "deployed" not in st.session_state \
                or st.session_state.get("cfg") != (n_start, seed):
            rng = np.random.default_rng(int(seed))
            st.session_state.deployed = [int(i) for i in rng.choice(len(df), n_start, replace=False)]
            st.session_state.cfg = (n_start, seed)
            st.session_state.recs = []
    else:
        if "planned" not in st.session_state:
            st.session_state.planned = []
        if st.button("Clear planned sensors", key="clear"):
            st.session_state.planned = []
            st.session_state.real_recs = []

    st.header("Chatbot")
    api_key = st.text_input("Gemini API key (optional)", type="password",
                            help="Without a key the chatbot uses simple keyword matching.")
    model_name = st.text_input("Gemini model", "gemini-2.5-flash")

if "recs" not in st.session_state:
    st.session_state.recs = []
if "real_recs" not in st.session_state:
    st.session_state.real_recs = []

# ---- compute everything for the chosen mode
if mode == "real":
    model = fit_model(kernel, X_all, y_all, p_all, S_HIST)          # every real sensor
    grid_mean, grid_std_now = predict(model, X_grid, grid["era5"].values, latent=True)
    planned = st.session_state.planned

    # uncertainty if the planned sensors were installed (no reading needed - only their location)
    Xk = np.vstack([X_all] + [X_grid[planned]]) if planned else X_all
    Fk = np.vstack([F_all] + [F_grid[planned]]) if planned else F_all
    grid_std = np.sqrt(uk_var(kernel, X_grid, Xk, F_grid, Fk, latent=True)) * S_HIST

    expected, z = check_readings(kernel, X_all, y_all, p_all, S_HIST)
    checks = df.copy()
    checks["expected"] = expected
    checks["z"] = z
    suspicious = checks[np.abs(checks.z) > 2].copy()
    suspicious = suspicious.reindex(suspicious.z.abs().sort_values(ascending=False).index)

    free = [i for i in range(len(grid)) if i not in planned]

    def recommend_fn(k):
        picks = greedy_pick(kernel, Xk, Fk, X_grid[free], F_grid[free], k, criterion)
        st.session_state.real_recs = [free[p] for p in picks]      # show them on the map too
        out = []
        for p in picks:
            g = free[p]
            out.append(describe_site(df, grid.latitude.iloc[g], grid.longitude.iloc[g]))
        return out

    summary = (f"The real network has **{len(df)} sensors**. Average uncertainty across the map is "
               f"**±{grid_std_now.mean():.2f}%**"
               + (f", dropping to **±{grid_std.mean():.2f}%** with the {len(planned)} planned sensor(s)" if planned else "")
               + f". **{len(suspicious)}** station(s) have readings that disagree strongly with their neighbours.")
    ctx = {"mode": mode, "df_known": df, "model": model, "criterion": criterion, "grid": grid,
           "grid_mean": grid_mean, "grid_std": grid_std, "suspicious": suspicious,
           "recommend_fn": recommend_fn, "summary": summary}
else:
    deployed = st.session_state.deployed
    unused = [i for i in range(len(df)) if i not in deployed]
    model = fit_model(kernel, X_all[deployed], y_all[deployed], p_all[deployed], S_HIST)
    grid_mean, grid_std = predict(model, X_grid, grid["era5"].values, latent=True)
    unused_mean, _ = predict(model, X_all[unused], p_all[unused])
    mae = float(np.mean(np.abs(unused_mean - y_all[unused]))) if unused else 0.0

    def recommend_fn(k):
        picks = greedy_pick(kernel, X_all[deployed], F_all[deployed], X_all[unused], F_all[unused], k, criterion)
        st.session_state.recs = [unused[p] for p in picks]         # show them on the map too
        out = []
        for p in picks:
            s = df.iloc[unused[p]]
            out.append(f"{s['name']} ({s['stateCode']}) — lat {s['latitude']:.2f}, lon {s['longitude']:.2f}")
        return out

    summary = (f"Simulation: pretending only **{len(deployed)} sensors** exist out of {len(df)}. "
               f"Average uncertainty is **±{grid_std.mean():.2f}%**, and the error at the {len(unused)} hidden "
               f"stations is **{mae:.2f}%** (MAE).")
    ctx = {"mode": mode, "df_known": df.iloc[deployed], "model": model, "criterion": criterion, "grid": grid,
           "grid_mean": grid_mean, "grid_std": grid_std, "suspicious": None,
           "recommend_fn": recommend_fn, "summary": summary}

# ---- top row
if mode == "real":
    st.info("**Real network** — uses every real sensor. Shows where sensors are missing, which readings look off, "
            "and where new sensors would help most.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Real sensors", len(df))
    c2.metric("Avg uncertainty", f"±{grid_std.mean():.2f}%",
              delta=f"{grid_std.mean() - grid_std_now.mean():.2f}% with planned" if planned else None,
              delta_color="inverse")
    c3.metric("Readings worth checking", len(suspicious))
else:
    st.info("**Proof (simulation)** — hides most real sensors, places them one by one, and checks the guesses "
            "against the hidden real readings. This is how the method was tested in the notebook.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensors in simulation", len(deployed))
    c2.metric("Avg uncertainty", f"±{grid_std.mean():.2f}%")
    c3.metric("Error at hidden stations", f"{mae:.2f}%")
st.caption(f"Calibration fitted on the sensors in use: station ≈ {model['a']:.1f} + {model['b']:.2f} × ERA5")

left, right = st.columns([3, 2])

with left:
    b1, b2 = st.columns(2)
    if mode == "real":
        if b1.button("Find best new sites", key="rec", type="primary", width="stretch"):
            st.session_state.real_recs = greedy_pick(kernel, Xk, Fk, X_grid[free], F_grid[free], k_rec, criterion)
            st.session_state.real_recs = [free[p] for p in st.session_state.real_recs]
        if b2.button("Add them as planned sensors", key="deploy", width="stretch",
                     disabled=not st.session_state.real_recs):
            st.session_state.planned = planned + st.session_state.real_recs
            st.session_state.real_recs = []
            st.rerun()
    else:
        if b1.button("Recommend next sensors", key="rec", type="primary", width="stretch"):
            picks = greedy_pick(kernel, X_all[deployed], F_all[deployed], X_all[unused], F_all[unused], k_rec, criterion)
            st.session_state.recs = [unused[p] for p in picks]
        if b2.button("Deploy recommended sensors", key="deploy", width="stretch",
                     disabled=not st.session_state.recs):
            st.session_state.deployed = deployed + st.session_state.recs
            st.session_state.recs = []
            st.rerun()

    layer = st.radio("Map layer", ["Uncertainty", "Predicted moisture", "Coarse ERA5 estimate"], horizontal=True, key="layer")
    if layer == "Uncertainty":
        values, colours, cap = grid_std, ["#fff5eb", "#fd8d3c", "#7f2704"], "Uncertainty about the true value (± % soil moisture)"
    elif layer == "Predicted moisture":
        values, colours, cap = grid_mean, ["#f7fcf0", "#7bccc4", "#084081"], "Predicted soil moisture (%)"
    else:
        values, colours, cap = grid["era5"].values, ["#f7fcf0", "#7bccc4", "#084081"], "ERA5-Land soil moisture (%)"
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9
    cmap = LinearColormap(colours, vmin=vmin, vmax=vmax, caption=cap)

    lons_u = np.sort(grid.longitude.unique())
    lats_u = np.sort(grid.latitude.unique())
    dlon = (lons_u[1] - lons_u[0]) / 2 if len(lons_u) > 1 else 1
    dlat = (lats_u[1] - lats_u[0]) / 2 if len(lats_u) > 1 else 1

    fmap = folium.Map(location=[39, -97], zoom_start=4, tiles="OpenStreetMap")
    for i in range(len(grid)):
        lat = grid.latitude.iloc[i]
        lon = grid.longitude.iloc[i]
        folium.Rectangle(bounds=[[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]],
                         color=None, weight=0, fill=True, fill_color=cmap(values[i]), fill_opacity=0.6,
                         tooltip=f"{grid_mean[i]:.1f}% ± {grid_std[i]:.1f}%").add_to(fmap)

    if mode == "real":
        sus_ids = set(suspicious.index)
        for i in range(len(df)):
            s = df.iloc[i]
            if i in sus_ids:
                folium.CircleMarker([s.latitude, s.longitude], radius=7, color="#e67e00", fill=True, fill_opacity=1,
                                    tooltip=f"CHECK: {s['name']} ({s['stateCode']}) reads {s['moisture']:.1f}%, "
                                            f"expected ~{checks.expected.iloc[i]:.1f}%").add_to(fmap)
            else:
                folium.CircleMarker([s.latitude, s.longitude], radius=4, color="#1f4e9c", fill=True, fill_opacity=1,
                                    tooltip=f"{s['name']} ({s['stateCode']}) — {s['moisture']:.1f}%").add_to(fmap)
        for g in planned:
            folium.Marker([grid.latitude.iloc[g], grid.longitude.iloc[g]], tooltip="Planned sensor",
                          icon=folium.Icon(color="purple", icon="plus")).add_to(fmap)
        for r in range(len(st.session_state.real_recs)):
            g = st.session_state.real_recs[r]
            folium.Marker([grid.latitude.iloc[g], grid.longitude.iloc[g]],
                          tooltip=f"#{r + 1} best new site",
                          icon=folium.Icon(color="red", icon="star")).add_to(fmap)
        legend = "Blue = real sensor · orange = reading worth checking · red star = best new site · purple = planned"
    else:
        for i in unused:
            s = df.iloc[i]
            folium.CircleMarker([s.latitude, s.longitude], radius=2, color="#888", fill=True,
                                tooltip=f"{s['name']} ({s['stateCode']}) — hidden").add_to(fmap)
        for i in deployed:
            s = df.iloc[i]
            folium.CircleMarker([s.latitude, s.longitude], radius=5, color="#1f4e9c", fill=True, fill_opacity=1,
                                tooltip=f"{s['name']} ({s['stateCode']}) — {s['moisture']:.1f}%").add_to(fmap)
        for r in range(len(st.session_state.recs)):
            s = df.iloc[st.session_state.recs[r]]
            folium.Marker([s.latitude, s.longitude], tooltip=f"#{r + 1}: {s['name']} ({s['stateCode']})",
                          icon=folium.Icon(color="red", icon="star")).add_to(fmap)
        legend = "Blue = sensor in the simulation · grey = hidden station · red star = recommended next sensor"

    cmap.add_to(fmap)
    st_folium(fmap, height=520, use_container_width=True, returned_objects=[])
    st.caption(legend)

    if mode == "real" and st.session_state.real_recs:
        st.markdown("**Best new sites for a sensor**")
        for r in range(len(st.session_state.real_recs)):
            g = st.session_state.real_recs[r]
            st.write(f"{r + 1}. {describe_site(df, grid.latitude.iloc[g], grid.longitude.iloc[g])}")
    if mode == "proof" and st.session_state.recs:
        st.markdown("**Recommended next sensors**")
        for r in range(len(st.session_state.recs)):
            s = df.iloc[st.session_state.recs[r]]
            st.write(f"{r + 1}. {s['name']} ({s['stateCode']}) — lat {s['latitude']:.2f}, lon {s['longitude']:.2f}")

    if mode == "real":
        with st.expander(f"Readings worth checking ({len(suspicious)})"):
            st.caption("Each station is compared with what its neighbours + the calibrated coarse estimate predict "
                       "without it. |z| > 2 means a bigger gap than normal noise explains — a faulty sensor, or "
                       "genuinely different local soil.")
            if len(suspicious):
                show = suspicious[["name", "stateCode", "moisture", "expected", "z"]].copy()
                show.columns = ["Station", "State", "Reading %", "Expected %", "z"]
                st.dataframe(show.round(2), hide_index=True, width="stretch")

with right:
    st.subheader("Ask the network")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    box = st.container(height=470)
    for role, text in st.session_state.chat:
        box.chat_message(role).markdown(text)
    q = st.chat_input("e.g. Where are we missing sensors?")
    if q:
        st.session_state.chat.append(("user", q))
        st.session_state.chat.append(("assistant", answer(q, ctx, api_key, model_name)))
        st.rerun()
    st.caption("The chatbot only picks which calculation to run — every number comes from the model, not the LLM.")
