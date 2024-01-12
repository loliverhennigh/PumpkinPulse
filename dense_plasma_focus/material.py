from dataclasses import dataclass
import enum

@dataclass
class Material:
    name: str
    color: str = "white"

VACUUM = Material(name="Vacuum", color="white")
COPPER = Material(name="Copper", color="orange")
QUARTZ = Material(name="Quartz", color="blue")
