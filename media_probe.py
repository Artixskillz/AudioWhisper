"""
Media probe — runs in the bundled Python runtime.
Prints JSON {duration: seconds, peaks: [0..1 floats]} for the GUI's
file info and waveform preview. Decodes at most the first 30 seconds.

PyAV bundles its own FFmpeg, so this works for audio and video alike.

Usage:
    python media_probe.py FILE [BARS]
"""
import sys
import json

PREVIEW_SECONDS = 30


def probe(path, bars):
    out = {"duration": 0.0, "peaks": []}

    import av
    import numpy as np

    with av.open(path) as container:
        # Container duration is in AV_TIME_BASE (microseconds)
        if container.duration:
            out["duration"] = container.duration / 1_000_000

        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return out

        if not out["duration"] and stream.duration and stream.time_base:
            out["duration"] = float(stream.duration * stream.time_base)

        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            chunks.append(np.abs(arr.astype(np.float32)))
            if frame.time is not None and frame.time >= PREVIEW_SECONDS:
                break

        if not chunks:
            return out

        y = np.concatenate(chunks)
        chunk_size = max(len(y) // bars, 1)
        peaks = [float(y[i * chunk_size:(i + 1) * chunk_size].max())
                 for i in range(min(bars, len(y) // chunk_size))]
        top = max(peaks) if peaks and max(peaks) > 0 else 1.0
        out["peaks"] = [round(p / top, 4) for p in peaks]

    return out


def main():
    path = sys.argv[1]
    bars = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    try:
        result = probe(path, bars)
    except Exception:
        result = {"duration": 0.0, "peaks": []}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
