"""Ask Gemini about images (or plain text) -- a cheap second pair of eyes.

Keys: one per line in gemini-keys.txt next to this file (local only, never
committed, never printed). Keys are tried in turn until one answers.

    python gemini.py "What do you see?" a.png b.png
    python gemini.py --model gemini-3.6-flash "question" shot.png
    python gemini.py --list            # model names this key can use
"""
import base64, io, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def keys():
    with open(os.path.join(HERE, "gemini-keys.txt"), encoding="utf-8") as f:
        return [k.strip() for k in f if k.strip()]


def call(url, key, body=None):
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def image_part(path, max_side=1280):
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        data, mime = buf.getvalue(), "image/jpeg"
    except ImportError:
        data, mime = open(path, "rb").read(), "image/png"
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}


def with_any_key(fn):
    errors = []
    for i, key in enumerate(keys()):
        try:
            return fn(key)
        except urllib.error.HTTPError as e:
            errors.append(f"key {i + 1}: HTTP {e.code} {e.read()[:200]!r}")
        except Exception as e:  # network, timeout
            errors.append(f"key {i + 1}: {e}")
    sys.exit("every key failed:\n  " + "\n  ".join(errors))


def main(argv):
    model = DEFAULT_MODEL
    if argv[:1] == ["--model"]:
        model, argv = argv[1], argv[2:]
    if argv[:1] == ["--list"]:
        out = with_any_key(lambda k: call(f"{API}/models?pageSize=200", k))
        for m in out.get("models", []):
            print(m["name"].removeprefix("models/"))
        return
    if not argv:
        sys.exit(__doc__)
    prompt, images = argv[0], argv[1:]
    body = {"contents": [{"parts": [{"text": prompt}] + [image_part(p) for p in images]}]}
    out = with_any_key(lambda k: call(f"{API}/models/{model}:generateContent", k, body))
    parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    print("".join(p.get("text", "") for p in parts).strip() or json.dumps(out)[:500])


if __name__ == "__main__":
    main(sys.argv[1:])
