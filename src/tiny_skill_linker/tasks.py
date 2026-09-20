"""The TechWolf ESCO skill-linking tasks this project evaluates on."""

from workrb.tasks.ranking.skill_extraction import (
    HouseSkillExtractRanking,
    TechSkillExtractRanking,
    TechWolfSkillExtractRanking,
)

TASKS = {
    "tech": TechSkillExtractRanking,
    "house": HouseSkillExtractRanking,
    "techwolf": TechWolfSkillExtractRanking,
}
# TECHWOLF ships a test split only.
VAL_TASKS = ("tech", "house")
ESCO_VERSION = "1.1.0"


def load_task(name: str, split: str):
    return TASKS[name](split=split, languages=["en"], esco_version=ESCO_VERSION)
