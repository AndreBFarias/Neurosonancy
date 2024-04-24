# -*- coding: utf-8 -*-

from textual.widgets import Static


class BentoBox(Static):

    DEFAULT_CSS = """
    BentoBox {
        border: solid #6272a4;
        background: #44475a;
        padding: 1;
        color: #f8f8f2;
    }

    BentoBox.critical {
        border: double #ff5555;
    }

    BentoBox.warning {
        border: heavy #ffb86c;
    }

    BentoBox.success {
        border: tall #50fa7b;
    }

    BentoBox.info {
        border: round #8be9fd;
    }

    BentoBox.accent {
        border: double #bd93f9;
    }
    """

    def __init__(
        self,
        content: str = "",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(content, name=name, id=id, classes=classes)
