# -*- coding: utf-8 -*-

from dataclasses import dataclass
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import ListView, ListItem, Static

from src.core.theme import COLORS


@dataclass
class NavItem:
    module_id: str
    label: str
    shortcut: str


_NAV_ITEMS: list[NavItem] = [
    NavItem("monitor", "Monitor", "1"),
    NavItem("trainer", "Trainer", "2"),
    NavItem("clone", "Clone Voice", "3"),
]


class NavSidebar(Widget):

    class ModuleSelected(Message):
        def __init__(self, module_id: str) -> None:
            super().__init__()
            self.module_id = module_id

    _STATUS_COLORS: dict[str, str] = {
        "REC": COLORS["neon_pink"],
        "GERANDO": COLORS["accent_primary"],
        "TREINANDO": COLORS["warning"],
    }

    DEFAULT_CSS = f"""
    NavSidebar {{
        width: 22;
        min-width: 16;
        height: 100%;
        background: {COLORS['bg_surface']};
        border-right: solid {COLORS['border_normal']};
    }}

    NavSidebar #sidebar-title {{
        text-align: center;
        color: {COLORS['accent_tertiary']};
        text-style: bold;
        padding: 1;
        background: {COLORS['bg_elevated']};
        border-bottom: solid {COLORS['border_normal']};
    }}

    NavSidebar ListView {{
        background: transparent;
        border: none;
        padding: 1 0;
    }}

    NavSidebar ListItem {{
        padding: 0 1;
        color: {COLORS['text_secondary']};
    }}

    NavSidebar ListItem:hover {{
        background: {COLORS['bg_hover']};
        color: {COLORS['text_primary']};
    }}

    NavSidebar ListItem.active {{
        color: {COLORS['accent_primary']};
        text-style: bold;
        background: {COLORS['bg_elevated']};
        border-left: thick {COLORS['accent_primary']};
        padding-left: 1;
    }}

    NavSidebar #sidebar-footer {{
        text-align: center;
        color: {COLORS['text_muted']};
        padding: 1;
        background: {COLORS['bg_elevated']};
        border-top: solid {COLORS['border_normal']};
        dock: bottom;
    }}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._module_statuses: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("NEUROSONANCY", id="sidebar-title")
        with ListView(id="nav-list"):
            for item in _NAV_ITEMS:
                label = f"[{item.shortcut}] {item.label}"
                li = ListItem(Static(label), id=f"nav-{item.module_id}")
                yield li
        yield Static("[T] Tray: OFF", id="sidebar-footer")

    def on_mount(self) -> None:
        self.set_active("monitor")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            module_id = event.item.id.replace("nav-", "")
            self.set_active(module_id)
            self.post_message(self.ModuleSelected(module_id))

    def set_active(self, module_id: str) -> None:
        for item in _NAV_ITEMS:
            try:
                li = self.query_one(f"#nav-{item.module_id}", ListItem)
                if item.module_id == module_id:
                    li.add_class("active")
                else:
                    li.remove_class("active")
            except Exception:
                pass

    def set_tray_status(self, running: bool) -> None:
        try:
            footer = self.query_one("#sidebar-footer", Static)
            state = "ON" if running else "OFF"
            footer.update(f"[T] Tray: {state}")
        except Exception:
            pass

    def set_module_status(self, module_id: str, status: str | None) -> None:
        """Atualiza badge de estado do item de menu. status=None remove o badge."""
        if status is None:
            self._module_statuses.pop(module_id, None)
        else:
            self._module_statuses[module_id] = status
        self._refresh_item_label(module_id)

    def _refresh_item_label(self, module_id: str) -> None:
        item = next((n for n in _NAV_ITEMS if n.module_id == module_id), None)
        if item is None:
            return
        base = f"[{item.shortcut}] {item.label}"
        status = self._module_statuses.get(module_id)
        if status:
            color = self._STATUS_COLORS.get(status, COLORS["text_muted"])
            label = f"{base} [{color}]{status}[/]"
        else:
            label = base
        try:
            li = self.query_one(f"#nav-{module_id}", ListItem)
            li.query_one(Static).update(label)
        except Exception:
            pass


# "Não procures que as coisas aconteçam como desejas; deseja que elas aconteçam como acontecem." — Epicteto
