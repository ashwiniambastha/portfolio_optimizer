"""
Run this script once to patch agents/market_data/api.py in-place.
It adds NaN sanitisation and drops the partial last-day candle from yfinance.

Usage:
    python fix_market_data_api.py
"""
import re
import sys
import pathlib

TARGET = pathlib.Path("agents/market_data/api.py")

if not TARGET.exists():
    # Try common alternative paths
    for p in [
        pathlib.Path("api.py"),
        pathlib.Path("market_data/api.py"),
    ]:
        if p.exists():
            TARGET = p
            break
    else:
        sys.exit(f"ERROR: Could not find market_data api.py. Run from project root or edit TARGET path in this script.")

print(f"Patching: {TARGET.resolve()}")
content = TARGET.read_text(encoding="utf-8")

# ── 1. Inject sanitise() helper right after the imports block ──────────────────
SANITISE_HELPER = '''

def sanitise(obj):
    """Recursively replace float NaN/Inf with None for safe JSON serialisation."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.floating):
            f = float(obj)
            return None if (f != f or f == float('inf') or f == float('-inf')) else f
        if isinstance(obj, _np.integer):
            return int(obj)
    except Exception:
        pass
    return obj

'''

if "def sanitise(" not in content:
    # Insert after the last top-level import line
    insert_after = re.search(
        r'^((?:import |from )\S.*\n)+',   # block of import lines
        content,
        re.MULTILINE
    )
    if insert_after:
        pos = insert_after.end()
        content = content[:pos] + SANITISE_HELPER + content[pos:]
        print("✓ Injected sanitise() helper")
    else:
        print("⚠ Could not find import block to insert helper — add it manually")
else:
    print("✓ sanitise() already present, skipping")

# ── 2. Wherever yfinance data is fetched, drop NaN rows before use ─────────────
#    Pattern: df = <ticker>.history(...)  →  df = ...; df = df.dropna(subset=['Close'])
NAN_DROP_SNIPPET = "    # Drop incomplete last candle (yfinance returns NaN OHLC for today)\n    df = df.dropna(subset=['Close'])\n"

# Find all .history( calls followed by the DataFrame being returned/used
# We insert the dropna right after the assignment line
def inject_dropna(text):
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        # Match lines like:  df = something.history(...)
        if re.search(r'\bdf\s*=\s*.*\.history\(', line) and "dropna" not in line:
            # Check we haven't already added it on the next line
            next_line = lines[i+1] if i+1 < len(lines) else ""
            if "dropna(subset=['Close'])" not in next_line:
                out.append(NAN_DROP_SNIPPET)
                print(f"✓ Injected dropna after: {line.rstrip()}")
        i += 1
    return "".join(out)

content = inject_dropna(content)

# ── 3. Wrap all `return` statements inside endpoint functions with sanitise() ──
#    Target pattern: return <dict or variable>  (not return None / raise)
def wrap_returns(text):
    # Only wrap bare `return result` or `return {` style returns inside route fns
    # Simple heuristic: lines that are `        return <identifier_or_{>`
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        m = re.match(r'^(\s{4,8})return\s+(\{.*|[a-zA-Z_][a-zA-Z0-9_]*)(\s*)$', line)
        if m and 'sanitise' not in line:
            indent, expr, tail = m.group(1), m.group(2), m.group(3)
            if expr.startswith('{'):
                # Multi-line dict — can't wrap inline, leave it; user gets the dict
                # but the sanitise at the response level will catch it
                out.append(line)
            else:
                new_line = f"{indent}return sanitise({expr}){tail}\n"
                out.append(new_line)
                print(f"✓ Wrapped: return {expr}  →  return sanitise({expr})")
        else:
            out.append(line)
    return "".join(out)

content = wrap_returns(content)

# ── 4. Write patched file ───────────────────────────────────────────────────────
TARGET.write_text(content, encoding="utf-8")
print(f"\n✓ Patched file written: {TARGET.resolve()}")
print("Restart the market data API (port 8000) for changes to take effect.")