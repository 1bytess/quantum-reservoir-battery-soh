from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_6_ROOT = PROJECT_ROOT / "result" / "phase_6"


@dataclass(frozen=True)
class Phase6StagePaths:
    stage_name: str

    @property
    def stage_dir(self) -> Path:
        return PHASE_6_ROOT / self.stage_name

    @property
    def data_dir(self) -> Path:
        return self.stage_dir / "data"

    @property
    def plot_dir(self) -> Path:
        return self.stage_dir / "plot"

    @property
    def hardware_dir(self) -> Path:
        return self.stage_dir / "hardware"

    @property
    def manifest_dir(self) -> Path:
        return self.hardware_dir / "manifest"

    @property
    def checkpoint_dir(self) -> Path:
        return self.hardware_dir / "checkpoint"

    def ensure_dirs(self, include_hardware: bool = False) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        if include_hardware:
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_stage_paths(stage_name: str, include_hardware: bool = False) -> Phase6StagePaths:
    paths = Phase6StagePaths(stage_name)
    paths.ensure_dirs(include_hardware=include_hardware)
    return paths
