
from app.core.permissions import PermissionManager, RiskLevel


def test_known_read_tool_is_allowed():
    permissions = PermissionManager()

    assert permissions.is_allowed("list_files") is True


def test_unknown_tool_is_denied():
    permissions = PermissionManager()

    assert permissions.is_allowed("delete_file") is False


def test_read_tool_has_correct_risk():
    permissions = PermissionManager()

    assert permissions.get_risk("read_file") == RiskLevel.READ


def test_unknown_tool_has_no_risk():
    permissions = PermissionManager()

    assert permissions.get_risk("unknown_tool") is None



def test_create_file_is_allowed():
    permissions = PermissionManager()
    assert permissions.get_risk("create_file") == RiskLevel.WRITE
    assert permissions.is_allowed("create_file") is True


def test_unknown_write_tool_is_denied():
    permissions = PermissionManager()
    assert permissions.is_allowed("delete_file") is False


def test_external_tool_is_denied():
    permissions = PermissionManager()
    permissions.register_tool("send_email", RiskLevel.EXTERNAL)
    assert permissions.is_allowed("send_email") is False


def test_registered_read_tool_is_allowed():
    permissions = PermissionManager()
    permissions.register_tool("analyze_document", RiskLevel.READ)
    assert permissions.is_allowed("analyze_document") is True


def test_registered_destructive_tool_is_blocked():
    permissions = PermissionManager()
    permissions.register_tool("delete_file", RiskLevel.DESTRUCTIVE)
    assert permissions.is_allowed("delete_file") is False
