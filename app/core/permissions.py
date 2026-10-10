
from enum import Enum


class RiskLevel(Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"


class PermissionManager:
    def __init__(self):
        self.tool_risks = {
            "get_current_directory": RiskLevel.READ,
            "list_files": RiskLevel.READ,
            "list_text_files": RiskLevel.READ,
            "read_file": RiskLevel.READ,
            "search_files": RiskLevel.READ,
            "create_file": RiskLevel.WRITE,
            "write_file": RiskLevel.WRITE,
            "validate_python_file": RiskLevel.READ,
        }

        self.allowed_risks = {
            RiskLevel.READ,
            RiskLevel.WRITE,
        }

    def get_risk(self, tool_name: str) -> RiskLevel | None:
        return self.tool_risks.get(tool_name)

    def is_allowed(self, tool_name: str) -> bool:
        risk = self.get_risk(tool_name)
        return risk is not None and risk in self.allowed_risks

    def register_tool(self, tool_name: str, risk: RiskLevel) -> None:
        if not tool_name.strip():
            raise ValueError("Tool name cannot be empty.")

        if not isinstance(risk, RiskLevel):
            raise TypeError("risk must be a RiskLevel value.")

        self.tool_risks[tool_name] = risk

    def unclassified_tools(self, tool_names: list[str]) -> list[str]:
        """Find discovered tools that have no explicit risk classification."""
        return sorted(set(tool_names) - set(self.tool_risks))
