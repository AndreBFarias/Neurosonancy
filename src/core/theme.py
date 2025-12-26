# -*- coding: utf-8 -*-

COLORS = {
    "bg_darkest": "#0d0f18",
    "bg_dark": "#141620",
    "bg_surface": "#1a1d2e",
    "bg_elevated": "#232738",
    "bg_hover": "#2a2f45",

    "border_dim": "#2d3250",
    "border_normal": "#3d4370",
    "border_accent": "#6366f1",

    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",

    "accent_primary": "#8b5cf6",
    "accent_secondary": "#6366f1",
    "accent_tertiary": "#a78bfa",

    "success": "#22c55e",
    "success_dim": "#15803d",
    "warning": "#f59e0b",
    "warning_dim": "#b45309",
    "error": "#ef4444",
    "error_dim": "#b91c1c",
    "info": "#06b6d4",
    "info_dim": "#0891b2",

    "neon_pink": "#ec4899",
    "neon_cyan": "#22d3ee",
    "neon_green": "#4ade80",
    "neon_orange": "#fb923c",
    "neon_purple": "#a855f7",
}

THEME = {
    "name": "Neurosonancy Dark",
    "version": "2.0",
    "font_mono": "JetBrains Mono, Fira Code, monospace",
    "border_radius": "6px",
    "transition": "0.15s ease",
}

CSS_COMMON = """
Screen {
    background: $bg_darkest;
}

Header {
    background: $bg_surface;
    color: $accent_primary;
    text-style: bold;
    height: 3;
    dock: top;
}

Header .header--icon {
    display: none;
}

Footer {
    background: $bg_surface;
    color: $text_secondary;
    height: 1;
}

Footer > .footer--key {
    background: $accent_primary;
    color: $bg_darkest;
    text-style: bold;
}

Footer > .footer--description {
    color: $text_secondary;
}

#app-title {
    text-align: center;
    width: 100%;
    color: $accent_primary;
    text-style: bold;
}

.panel {
    background: $bg_surface;
    border: solid $border_dim;
    padding: 1 2;
}

.panel-title {
    color: $accent_tertiary;
    text-style: bold;
    text-align: center;
    padding-bottom: 1;
    border-bottom: solid $border_dim;
    margin-bottom: 1;
}

.card {
    background: $bg_elevated;
    border: solid $border_dim;
    padding: 1 2;
    margin: 1;
}

.card:hover {
    border: solid $accent_primary;
    background: $bg_hover;
}

.card-header {
    color: $text_primary;
    text-style: bold;
    padding-bottom: 1;
}

.card-body {
    color: $text_secondary;
}

Button {
    background: $bg_elevated;
    color: $text_primary;
    border: solid $border_normal;
    height: 3;
    min-width: 12;
    text-style: bold;
}

Button:hover {
    background: $accent_primary;
    color: $bg_darkest;
    border: solid $accent_primary;
}

Button:focus {
    border: solid $accent_tertiary;
}

Button.-primary {
    background: $accent_primary;
    color: $bg_darkest;
    border: none;
}

Button.-primary:hover {
    background: $accent_tertiary;
}

Button.-success {
    background: $success;
    color: $bg_darkest;
    border: none;
}

Button.-success:hover {
    background: $success_dim;
    color: $text_primary;
}

Button.-warning {
    background: $warning;
    color: $bg_darkest;
    border: none;
}

Button.-warning:hover {
    background: $warning_dim;
    color: $text_primary;
}

Button.-error {
    background: $error;
    color: $bg_darkest;
    border: none;
}

Button.-error:hover {
    background: $error_dim;
    color: $text_primary;
}

Button:disabled {
    background: $bg_surface;
    color: $text_muted;
    border: solid $border_dim;
}

Input {
    background: $bg_dark;
    color: $text_primary;
    border: solid $border_dim;
    padding: 0 1;
}

Input:focus {
    border: solid $accent_primary;
}

Input.-invalid {
    border: solid $error;
}

ProgressBar {
    height: 1;
    padding: 0;
}

ProgressBar > .bar--bar {
    color: $border_dim;
}

ProgressBar > .bar--complete {
    color: $accent_primary;
}

ProgressBar.-success > .bar--complete {
    color: $success;
}

RichLog {
    background: $bg_dark;
    border: solid $border_dim;
    padding: 1;
    scrollbar-background: $bg_surface;
    scrollbar-color: $border_normal;
    scrollbar-color-hover: $accent_primary;
}

RadioSet {
    background: $bg_dark;
    border: solid $border_dim;
    padding: 1;
}

RadioButton {
    background: transparent;
    color: $text_secondary;
}

RadioButton:hover {
    color: $text_primary;
}

RadioButton.-selected {
    color: $accent_primary;
    text-style: bold;
}

Static.status-ok {
    color: $success;
}

Static.status-warning {
    color: $warning;
}

Static.status-error {
    color: $error;
}

Static.status-info {
    color: $info;
}

.neon-pink { color: $neon_pink; }
.neon-cyan { color: $neon_cyan; }
.neon-green { color: $neon_green; }
.neon-orange { color: $neon_orange; }
.neon-purple { color: $neon_purple; }
.text-muted { color: $text_muted; }
.text-primary { color: $text_primary; }
.text-secondary { color: $text_secondary; }
.text-accent { color: $accent_primary; }

.nav-hint {
    color: $text_muted;
    text-align: center;
    padding: 1;
}

.module-header {
    height: 5;
    background: $bg_surface;
    border-bottom: solid $border_dim;
    padding: 1 2;
}

.module-header-title {
    text-align: center;
    color: $accent_primary;
    text-style: bold;
}

.module-header-subtitle {
    text-align: center;
    color: $text_muted;
}

.section-divider {
    border-top: solid $border_dim;
    margin: 1 0;
    height: 1;
}
""".replace("$bg_darkest", COLORS["bg_darkest"]) \
   .replace("$bg_dark", COLORS["bg_dark"]) \
   .replace("$bg_surface", COLORS["bg_surface"]) \
   .replace("$bg_elevated", COLORS["bg_elevated"]) \
   .replace("$bg_hover", COLORS["bg_hover"]) \
   .replace("$border_dim", COLORS["border_dim"]) \
   .replace("$border_normal", COLORS["border_normal"]) \
   .replace("$border_accent", COLORS["border_accent"]) \
   .replace("$text_primary", COLORS["text_primary"]) \
   .replace("$text_secondary", COLORS["text_secondary"]) \
   .replace("$text_muted", COLORS["text_muted"]) \
   .replace("$accent_primary", COLORS["accent_primary"]) \
   .replace("$accent_secondary", COLORS["accent_secondary"]) \
   .replace("$accent_tertiary", COLORS["accent_tertiary"]) \
   .replace("$success_dim", COLORS["success_dim"]) \
   .replace("$success", COLORS["success"]) \
   .replace("$warning_dim", COLORS["warning_dim"]) \
   .replace("$warning", COLORS["warning"]) \
   .replace("$error_dim", COLORS["error_dim"]) \
   .replace("$error", COLORS["error"]) \
   .replace("$info_dim", COLORS["info_dim"]) \
   .replace("$info", COLORS["info"]) \
   .replace("$neon_pink", COLORS["neon_pink"]) \
   .replace("$neon_cyan", COLORS["neon_cyan"]) \
   .replace("$neon_green", COLORS["neon_green"]) \
   .replace("$neon_orange", COLORS["neon_orange"]) \
   .replace("$neon_purple", COLORS["neon_purple"])
