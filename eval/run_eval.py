import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from scorer import score_question

load_dotenv()

HERE = Path(__file__).parent
TESTSET_PATH = HERE / "testset.json"
REPORTS_DIR = HERE / "reports"

ENDPOINT = os.environ.get("EVAL_ENDPOINT")      # your API Gateway URL
SECRET = os.environ.get("EVAL_APP_SECRET")      # same value as the Lambda's APP_SECRET
DELAY_SECONDS = 3                               # stay under API Gateway throttle + Gemini free-tier limits


def ask(question, history=None, retries=3):
    """Returns (response_json, None) on success or (None, error_string) on failure."""
    payload = {"question": question}
    if history:
        payload["history"] = history

    err = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.post(
                ENDPOINT,
                json=payload,
                headers={"X-App-Secret": SECRET},
                timeout=35,
            )
        except requests.RequestException as e:
            err = str(e)
        else:
            if r.status_code == 200:
                return r.json(), None
            err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                return None, err          # e.g. 403 or 400: retrying won't help
        time.sleep(2 ** attempt * 2)
    return None, err


def summarize(results):
    cats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0, "unjudged": 0, "error": 0,
                                "src_hits": 0, "src_checked": 0, "scores": []})
    for r in results:
        c = cats[r["category"]]
        c["total"] += 1
        if "error" in r:
            c["error"] += 1
            continue
        if r["judge_pass"] is True:
            c["pass"] += 1
        elif r["judge_pass"] is False:
            c["fail"] += 1
        else:
            c["unjudged"] += 1
        if r["source_match"] is not None:
            c["src_checked"] += 1
            c["src_hits"] += int(r["source_match"])
        if r["top_score"] is not None:
            c["scores"].append(r["top_score"])
    return cats


def print_summary(results, cats):
    print("\n" + "=" * 72)
    print(f"{'category':<22}{'pass':>5}{'fail':>6}{'?':>4}{'err':>5}   {'source':<8}top_score range")
    for name, c in cats.items():
        src = f"{c['src_hits']}/{c['src_checked']}" if c["src_checked"] else "-"
        rng = f"{min(c['scores']):.2f}-{max(c['scores']):.2f}" if c["scores"] else "none"
        print(f"{name:<22}{c['pass']:>5}{c['fail']:>6}{c['unjudged']:>4}{c['error']:>5}   {src:<8}{rng}")
    total = sum(c["total"] for c in cats.values())
    passed = sum(c["pass"] for c in cats.values())
    print(f"\nOverall: {passed}/{total} passed")

    print("\nProblems:")
    for r in results:
        if "error" in r:
            print(f"  ERROR {r['id']}: {r['error']}")
        elif r["judge_pass"] is not True:
            print(f"  FAIL  {r['id']}: {r['question'][:60]}")
            print(f"        answer: {r['answer'][:150]}")
            print(f"        judge:  {r['judge_reason']}")


def run(label):
    if not ENDPOINT or not SECRET:
        sys.exit("Set EVAL_ENDPOINT and EVAL_APP_SECRET in your .env first.")

    testset = json.loads(TESTSET_PATH.read_text())
    results = []

    for i, q in enumerate(testset, 1):
        print(f"[{i}/{len(testset)}] {q['id']} ({q['category']}): {q['question'][:60]}")
        data, err = ask(q["question"], q.get("history"))

        if err:
            results.append({"id": q["id"], "category": q["category"],
                            "question": q["question"], "error": err})
        else:
            sources = data.get("sources", [])
            scored = score_question(q, data["answer"], sources)
            scored["answer"] = data["answer"]
            scored["sources"] = [s["title"] for s in sources]
            scored["top_score"] = max((s["score"] for s in sources), default=None)
            results.append(scored)

        time.sleep(DELAY_SECONDS)

    cats = summarize(results)
    print_summary(results, cats)

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.json"
    out.write_text(json.dumps({"label": label, "summary": cats, "results": results}, indent=2))
    print(f"\nSaved report: {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "run")
