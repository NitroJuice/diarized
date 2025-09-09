#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from speechbrain.inference import SpeakerRecognition

# ─────────────────────────────── logging ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────── helpers ────────────────────────────────
TMP_SR = 16_000
MIN_SAMPLES = TMP_SR // 10  # 100 ms


def run(cmd: list[str]) -> None:
    """Thin wrapper around subprocess.run with sanity logging."""
    log.debug("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────── pipeline ───────────────────────────────
class Diarizer:
    def __init__(self, whisper_bin: str = "whisper"):
        self.whisper_bin = whisper_bin
        self.spk_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        run(["ffmpeg", "-version"])  # fail-fast if ffmpeg absent

    # --------------------------- whisper ---------------------------
    def transcribe(self, audio: Path) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / (audio.stem + ".json")
            cmd = [
                self.whisper_bin,
                str(audio),
                "--model",
                "turbo",
                "--output_format",
                "json",
                "--output_dir",
                tmp,
                "--threads",
                str(os.cpu_count() or 1),
                "--verbose",
                "False",
            ]
            if torch.cuda.is_available():
                cmd += ["--device", "cuda"]
            run(cmd)

            if not out_json.exists():  # whisper sometimes names file oddly
                files = list(Path(tmp).glob("*.json"))
                if not files:
                    raise RuntimeError("Whisper produced no JSON")
                out_json = files[0]

            return json.loads(out_json.read_text(encoding="utf-8"))

    # --------------------------- audio ----------------------------
    def load_audio(self, audio: Path) -> torch.Tensor:
        if audio.suffix.lower() == ".mp3":
            wav = Path(tempfile.mktemp(suffix=".wav"))
            run(["ffmpeg", "-i", str(audio), "-ar", str(TMP_SR), "-ac", "1", "-y", str(wav)])
            path = wav
        else:
            path = audio

        signal, _ = librosa.load(path, sr=TMP_SR, mono=True)
        if path is not audio:
            path.unlink(missing_ok=True)
        return torch.from_numpy(signal).float().unsqueeze(0)  # (1, N)

    # ------------------------- embeddings -------------------------
    @torch.inference_mode()
    def extract_embeddings(self, signal: torch.Tensor, segments: list[dict]) -> tuple[np.ndarray, List[dict]]:
        embs, valid = [], []
        for seg in segments:
            s, e = int(seg["start"] * TMP_SR), int(seg["end"] * TMP_SR)
            if e - s < MIN_SAMPLES or e > signal.shape[1]:
                continue
            emb = self.spk_model.encode_batch(signal[:, s:e]).squeeze().cpu().numpy()
            embs.append(emb)
            valid.append(seg)
        return np.vstack(embs), valid

    # -------------------------- clustering ------------------------
    def choose_k(self, X: np.ndarray, k_user: int | None) -> int:
        if k_user:
            return max(1, min(k_user, len(X)))
        if len(X) < 3:
            return len(X)
        scores = []
        for k in range(2, min(10, len(X)) + 1):
            lbl = KMeans(k, n_init="auto", random_state=42).fit_predict(X)
            scores.append(silhouette_score(X, lbl))
        return scores.index(max(scores)) + 2

    def cluster(self, X: np.ndarray, k: int) -> np.ndarray:
        return KMeans(k, n_init="auto", random_state=42).fit_predict(
            StandardScaler().fit_transform(X)
        )

    # ---------------------------- api -----------------------------
    def diarize(self, audio: Path, exp_spk: int | None = None) -> dict:
        t0 = time.time()
        whisper_res = self.transcribe(audio)
        signal = self.load_audio(audio)
        X, valid_segs = self.extract_embeddings(signal, whisper_res["segments"])
        if X.size == 0:  # fallback
            for s in whisper_res["segments"]:
                s["speaker"] = "speaker_0"
            return whisper_res

        k = self.choose_k(X, exp_spk)
        labels = self.cluster(X, k)
        for seg, lab in zip(valid_segs, labels):
            seg["speaker"] = f"speaker_{lab}"
        log.info("Finished diarisation in %.1fs (k=%d)", time.time() - t0, k)
        return whisper_res


# ─────────────────────────────── cli ────────────────────────────────
def to_txt(res: dict) -> str:
    lines = []
    for idx, seg in enumerate(res["segments"], 1):
        lines.append(
            f"[{idx:03}] {seg.get('speaker','spk')} "
            f"({seg['start']:.1f}-{seg['end']:.1f}s): {seg['text'].strip()}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Diarise an audio file with Whisper + SpeechBrain")
    p.add_argument("audio", type=Path, help="Audio file (wav/mp3)")
    p.add_argument("-s", "--speakers", type=int, help="Expected speaker count")
    p.add_argument("-o", "--output", type=Path, help="Text output file")
    args = p.parse_args()

    if not args.audio.is_file():
        sys.exit(f"File not found: {args.audio}")

    out = args.output or Path.cwd() / (args.audio.stem + ".txt")

    diarizer = Diarizer()
    result = diarizer.diarize(args.audio, args.speakers)

    out.write_text(to_txt(result), encoding="utf-8")
    log.info("Results written to %s", out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
