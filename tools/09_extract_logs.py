# ================================================================
# 09_extract_logs: collects the semi-synthetic check lines from the saved outputs of all notebooks in this folder
# Save the notebooks first: outputs are read from the .ipynb files.
# ================================================================
import json, glob

KEYS = ("SEMI-SYNTHÉTIQUE (rotation)", "SEMI-SYNTHÉTIQUE (case-mix pur)")
seen = set()
for NB in sorted(glob.glob("*.ipynb")):
    with open(NB, encoding="utf-8") as fh:
        nb = json.load(fh)
    print(f"\n--- {NB} ---")
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                print(f"ERREUR : {out.get('ename')}: {out.get('evalue')}")
                continue
            text = out.get("text") or out.get("data", {}).get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
            for line in text.splitlines():
                if any(k in line for k in KEYS) and line.strip() not in seen:
                    seen.add(line.strip())
                    print(line.strip())
print(f"\n✓ {len(seen)} lignes SEMI-SYNTHÉTIQUE différentes trouvées (attendu : 6)")
