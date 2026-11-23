"""Phase 1 entry point: python -m src.pipelines.run_phase1"""
from src.data.inspection import run_inspection

if __name__ == "__main__":
    rep = run_inspection()
    print("QoL participants:", rep["qol_stress"]["n_participants"])
    print("MMASH folders:", rep["mmash"]["n_participant_folders"])
    print("Written: results/reports/phase1_inspection.json")
