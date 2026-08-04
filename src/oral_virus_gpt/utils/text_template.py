from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CLINICAL_DIAGNOSTIC_V1: Final[str] = (
    "Patient demographics: {demographics}\n"
    "Chief complaint: {chief_complaint}\n"
    "Medical history: {history}\n"
    "Clinical findings: {findings}\n"
    "Provide a differential diagnosis with calibrated confidence."
)


@dataclass(slots=True)
class ClinicalRecord:
    demographics: str
    chief_complaint: str
    history: str
    findings: str

    def render(self, template: str = CLINICAL_DIAGNOSTIC_V1) -> str:
        return template.format(
            demographics=self.demographics or "n/a",
            chief_complaint=self.chief_complaint or "n/a",
            history=self.history or "n/a",
            findings=self.findings or "n/a",
        )
