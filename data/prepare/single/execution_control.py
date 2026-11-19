"""Check manual pauses and disabled methods before starting a benchmark run."""
from pathlib import Path
import json
R=Path('@STUDY@/uc_fig2_benchmark_20260911')
def assert_can_start(method):
 p=R/'manifests/PAUSED.json'
 if p.exists():raise SystemExit('BENCHMARK PAUSED: explicit resume and checkpoint reconciliation required')
 p=R/'manifests/DISABLED_METHODS.json'
 if p.exists() and method in json.loads(p.read_text()).get('methods',[]):raise SystemExit('METHOD DISABLED: '+method)
