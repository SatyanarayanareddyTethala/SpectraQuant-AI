#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from perplexity import Perplexity


@dataclass
class ResearchSource:
    url: str = ""


@dataclass
class ResearchResult:
    query: str
    model: str
    created_at_utc: str
    output_text: str
    citations: List[str] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] or "research"


def ensure_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
    if not api_key:
        raise RuntimeError(
            'Missing PERPLEXITY_API_KEY or PPLX_API_KEY.\n'
            'Set one of them first, for example:\n'
            'export PERPLEXITY_API_KEY="your_rotated_key"'
        )
    return api_key


def save_markdown(result: ResearchResult, path: Path) -> None:
    lines: List[str] = [
        "# Research Report",
        "",
        f"**Query:** {result.query}",
        f"**Model:** {result.model}",
        f"**Created (UTC):** {result.created_at_utc}",
        "",
        "## Answer",
        "",
        result.output_text.strip(),
        "",
    ]

    if result.citations:
        lines.extend(["## Citations", ""])
        for i, url in enumerate(result.citations, 1):
            lines.append(f"{i}. {url}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_research_prompt(user_query: str, mode: str, region: Optional[str], output_style: str) -> str:
    parts = [
        "You are a research-grade analyst performing web-grounded synthesis.",
        "",
        "Requirements:",
        "1. Use current web information.",
        "2. Distinguish facts, estimates, and inferences explicitly.",
        "3. Prefer primary or high-credibility sources when available.",
        "4. When sources conflict, say so.",
        "5. Be concrete and decision-useful.",
        "",
        f"Analysis mode: {mode}",
        f"Output style: {output_style}",
    ]
    if region:
        parts.append(f"Regional focus: {region}")

    parts.extend([
        "",
        "Return this structure:",
        "A. Executive summary",
        "B. Key findings",
        "C. Evidence and comparisons",
        "D. Risks / caveats",
        "E. Practical conclusion",
        "",
        f"User query: {user_query}",
    ])
    return "\n".join(parts)


class PerplexityResearchAgent:
    def __init__(self, api_key: str) -> None:
        self.client = Perplexity(api_key=api_key)

    def create_completion(
        self,
        *,
        model: str,
        prompt: str,
        retries: int = 4,
        retry_base_seconds: float = 1.5,
    ):
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise, web-grounded research analyst."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )
            except Exception as exc:
                last_error = exc
                if attempt == retries:
                    break
                sleep_for = retry_base_seconds * (2 ** (attempt - 1))
                print(f"Attempt {attempt}/{retries} failed: {exc}")
                print(f"Retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)

        raise RuntimeError(f"Perplexity request failed after {retries} attempts: {last_error}") from last_error

    def run_research(
        self,
        *,
        query: str,
        model: str,
        mode: str,
        region: Optional[str],
        output_style: str,
    ) -> ResearchResult:
        prompt = build_research_prompt(
            user_query=query,
            mode=mode,
            region=region,
            output_style=output_style,
        )

        completion = self.create_completion(model=model, prompt=prompt)

        raw = completion.model_dump() if hasattr(completion, "model_dump") else json.loads(
            json.dumps(completion, default=lambda o: getattr(o, "__dict__", str(o)))
        )

        output_text = completion.choices[0].message.content
        citations = raw.get("citations", []) if isinstance(raw, dict) else []

        return ResearchResult(
            query=query,
            model=model,
            created_at_utc=utc_now_iso(),
            output_text=output_text or "",
            citations=citations,
            raw_response=raw,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perplexity research agent")
    parser.add_argument("--query", required=True, help="Research query")
    parser.add_argument("--model", default="sonar-pro", help="Model name")
    parser.add_argument("--mode", default="decision-grade comparative research")
    parser.add_argument("--region", default=None)
    parser.add_argument("--output-style", default="dense analytical memo")
    parser.add_argument("--outdir", default="research_runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        api_key = ensure_api_key()
        agent = PerplexityResearchAgent(api_key=api_key)

        result = agent.run_research(
            query=args.query,
            model=args.model,
            mode=args.mode,
            region=args.region,
            output_style=args.output_style,
        )

        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(args.query)}"
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        md_path = outdir / f"{run_id}.md"
        json_path = outdir / f"{run_id}.json"

        save_markdown(result, md_path)
        json_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== ANSWER ===\n")
        print(result.output_text)

        if result.citations:
            print("\n=== CITATIONS ===\n")
            for i, url in enumerate(result.citations, 1):
                print(f"{i}. {url}")

        print(f"\nSaved markdown: {md_path}")
        print(f"Saved json: {json_path}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
