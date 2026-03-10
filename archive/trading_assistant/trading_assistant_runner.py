import os, subprocess, datetime, json
from pathlib import Path
import smtplib
from email.message import EmailMessage

REPO = Path(__file__).resolve().parent
STATE = REPO / "data" / "assistant_state.json"
STATE.parent.mkdir(parents=True, exist_ok=True)

def sh(cmd: str):
    print(">>", cmd)
    return subprocess.run(cmd, shell=True, check=True, cwd=str(REPO), capture_output=True, text=True).stdout

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"last_plan_date": None, "last_hourly": None}

def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))

def send_email(subject: str, body: str):
    # Use Gmail App Password (recommended) OR your SMTP provider
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ["SMTP_USER"]
    SMTP_PASS = os.environ["SMTP_PASS"]
    TO_EMAIL   = os.environ["TO_EMAIL"]

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def latest_file(glob_pat: str):
    files = sorted(Path("reports").glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def build_plan():
    # News scan + pipeline based on news universe
    sh("spectraquant news-scan")
    sh("spectraquant download --from-news")
    sh("spectraquant predict")
    sh("spectraquant signals")
    sh("spectraquant portfolio")

    sig = latest_file("signals/top_signals_*.csv")
    cand = latest_file("news/news_candidates_*.csv")
    met = Path("reports/portfolio/portfolio_metrics.json")
    wts = Path("reports/portfolio/portfolio_weights.csv")

    parts = []
    parts.append("=== SpectraQuant Daily Plan (News-First) ===\n")
    if cand: parts.append(f"News candidates: {cand}\n")
    if sig:  parts.append(f"Top signals: {sig}\n")
    if wts.exists(): parts.append(f"Portfolio weights: {wts}\n")
    if met.exists(): parts.append(f"Portfolio metrics: {met}\n")
    parts.append("\nTip: If weights are empty, reduce alpha threshold or adjust buy/sell thresholds.\n")
    return "".join(parts)

def hourly_update():
    # Lightweight: just news scan + candidates + (optional) signals refresh without full download
    sh("spectraquant news-scan")
    cand = latest_file("news/news_candidates_*.csv")

    body = ["=== Hourly News Update (News-First) ===\n"]
    if cand:
        body.append(f"Latest candidates file: {cand}\n")
    else:
        body.append("No candidates found.\n")
    return "".join(body)

def main():
    mode = os.environ.get("ASSISTANT_MODE", "premarket")  # premarket | hourly
    now = datetime.datetime.now()
    today = now.date().isoformat()

    st = load_state()

    if mode == "premarket":
        if st.get("last_plan_date") == today:
            print("Already sent today's plan.")
            return
        body = build_plan()
        send_email(f"SpectraQuant Plan — {today}", body)
        st["last_plan_date"] = today
        save_state(st)
        print("✅ Sent premarket plan email.")

    elif mode == "hourly":
        # avoid duplicate sends within same hour
        hour_key = now.strftime("%Y-%m-%d %H:00")
        if st.get("last_hourly") == hour_key:
            print("Already sent this hour.")
            return
        body = hourly_update()
        send_email(f"SpectraQuant Hourly Update — {hour_key}", body)
        st["last_hourly"] = hour_key
        save_state(st)
        print("✅ Sent hourly update email.")
    else:
        raise SystemExit("Unknown ASSISTANT_MODE. Use premarket or hourly.")

if __name__ == "__main__":
    main()
