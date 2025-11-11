"""Gradio UI builder from extension manifests.

Reads manifest.json and generates Gradio components dynamically.
"""

import gradio as gr
import requests
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


class UIBuilder:
    """Builds Gradio UI components from extension manifests."""

    def __init__(self, extension_manager, daemon_client):
        """Initialize UI builder.

        Args:
            extension_manager: ExtensionManager instance
            daemon_client: DaemonClient instance
        """
        self.extension_manager = extension_manager
        self.daemon_client = daemon_client
        self.component_registry: Dict[str, Any] = {}

    def build_extension_tab(self, extension) -> gr.Tab:
        """Build a complete tab for an extension.

        Args:
            extension: Extension object

        Returns:
            gr.Tab component
        """
        ext_name = extension.manifest.get("extension", {}).get("name", extension.name)

        with gr.Tab(ext_name) as tab:
            # Get sidebar panel config
            sidebar_config = extension.manifest.get("sidebar_panel", {})
            display_config = extension.manifest.get("display", {})

            # Layout: video/display on left, controls on right
            with gr.Row():
                # Left column: Display (if extension has one)
                if extension.has_display:
                    with gr.Column(scale=2):
                        self._build_display(extension, display_config)

                # Right column: Controls
                with gr.Column(scale=1):
                    self._build_controls(extension, sidebar_config)

        return tab

    def _build_display(self, extension, display_config: Dict[str, Any]) -> None:
        """Build display component (video/HTML).

        Args:
            extension: Extension object
            display_config: Display configuration from manifest
        """
        title = display_config.get("title", "Display")
        description = display_config.get("description", "")

        if title:
            gr.Markdown(f"## {title}")
        if description:
            gr.Markdown(f"*{description}*")

        # Embed extension's display HTML via iframe or direct HTML
        display_url = extension.display_url
        if display_url:
            # Use HTML component with iframe
            gr.HTML(f"""
                <iframe
                    src="{display_url}"
                    style="width:100%; height:600px; border:1px solid #ccc; border-radius:8px;"
                    frameborder="0">
                </iframe>
            """)

    def _build_controls(self, extension, sidebar_config: Dict[str, Any]) -> None:
        """Build control panel from sidebar config.

        Args:
            extension: Extension object
            sidebar_config: Sidebar panel configuration from manifest
        """
        title = sidebar_config.get("title", "Controls")
        icon = sidebar_config.get("icon", "")

        if title:
            gr.Markdown(f"## {icon} {title}" if icon else f"## {title}")

        controls = sidebar_config.get("controls", [])
        for control in controls:
            self._build_control(extension, control)

    def _build_control(self, extension, control: Dict[str, Any]) -> Optional[Any]:
        """Build a single control component.

        Args:
            extension: Extension object
            control: Control configuration

        Returns:
            Gradio component or None
        """
        control_type = control.get("type")
        control_id = control.get("id", "")

        if control_type == "button":
            return self._build_button(extension, control)
        elif control_type == "slider":
            return self._build_slider(extension, control)
        elif control_type == "dropdown":
            return self._build_dropdown(extension, control)
        elif control_type == "toggle":
            return self._build_toggle(extension, control)
        elif control_type == "status_text":
            return self._build_status_text(extension, control)
        elif control_type == "section_header":
            return self._build_section_header(control)
        elif control_type == "separator":
            return gr.HTML("<hr>")
        elif control_type == "text_area":
            return self._build_text_area(extension, control)
        else:
            logger.warning(f"Unknown control type: {control_type}")
            return None

    def _build_button(self, extension, control: Dict[str, Any]) -> gr.Button:
        """Build button component."""
        label = control.get("label", "Button")
        endpoint = control.get("endpoint", "")
        method = control.get("method", "POST")
        color = control.get("color", "secondary")

        # Map color to Gradio variant
        variant_map = {
            "green": "primary",
            "red": "stop",
            "blue": "secondary",
            "gray": "secondary"
        }
        variant = variant_map.get(color, "secondary")

        btn = gr.Button(label, variant=variant)

        # Wire up click handler
        if endpoint:
            def on_click():
                try:
                    url = f"{extension.api_base_url}{endpoint}"
                    response = requests.request(method, url, timeout=2)
                    if response.status_code == 200:
                        return f"✓ {label} succeeded"
                    else:
                        return f"✗ {label} failed ({response.status_code})"
                except Exception as e:
                    return f"✗ Error: {str(e)[:50]}"

            # Create status textbox for feedback
            status = gr.Textbox(label="Status", interactive=False, visible=True)
            btn.click(fn=on_click, inputs=[], outputs=[status])

        return btn

    def _build_slider(self, extension, control: Dict[str, Any]) -> gr.Slider:
        """Build slider component."""
        label = control.get("label", "Slider")
        minimum = control.get("min", 0)
        maximum = control.get("max", 100)
        default = control.get("default", (minimum + maximum) / 2)
        step = control.get("step", 1)
        endpoint = control.get("endpoint", "")
        method = control.get("method", "POST")

        slider = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            value=default,
            step=step,
            label=label
        )

        # Wire up change handler
        if endpoint:
            def on_change(value):
                try:
                    url = f"{extension.api_base_url}{endpoint}"
                    payload = {"value": value}
                    response = requests.request(method, url, json=payload, timeout=2)
                    return value
                except Exception as e:
                    logger.error(f"Slider change error: {e}")
                    return value

            slider.change(fn=on_change, inputs=[slider], outputs=[slider])

        return slider

    def _build_dropdown(self, extension, control: Dict[str, Any]) -> gr.Dropdown:
        """Build dropdown component."""
        label = control.get("label", "Dropdown")
        options = control.get("options", [])
        default = control.get("default", None)
        endpoint = control.get("endpoint", "")
        method = control.get("method", "POST")

        # Extract choices from options
        choices = [opt.get("label", opt.get("value", "")) for opt in options]
        values = [opt.get("value", opt.get("label", "")) for opt in options]

        dropdown = gr.Dropdown(
            choices=choices,
            value=default or (choices[0] if choices else None),
            label=label
        )

        # Wire up change handler
        if endpoint:
            def on_change(value):
                try:
                    # Find the actual value corresponding to label
                    idx = choices.index(value) if value in choices else 0
                    actual_value = values[idx]

                    url = f"{extension.api_base_url}{endpoint}"
                    payload = {"value": actual_value}
                    response = requests.request(method, url, json=payload, timeout=2)
                    return value
                except Exception as e:
                    logger.error(f"Dropdown change error: {e}")
                    return value

            dropdown.change(fn=on_change, inputs=[dropdown], outputs=[dropdown])

        return dropdown

    def _build_toggle(self, extension, control: Dict[str, Any]) -> gr.Checkbox:
        """Build toggle/checkbox component."""
        label = control.get("label", "Toggle")
        default = control.get("default", False)
        endpoint = control.get("endpoint", "")
        method = control.get("method", "POST")
        description = control.get("description", "")

        checkbox = gr.Checkbox(
            value=default,
            label=label,
            info=description if description else None
        )

        # Wire up change handler
        if endpoint:
            def on_change(value):
                try:
                    url = f"{extension.api_base_url}{endpoint}"
                    payload = {"enabled": value}
                    response = requests.request(method, url, json=payload, timeout=2)
                    return value
                except Exception as e:
                    logger.error(f"Checkbox change error: {e}")
                    return value

            checkbox.change(fn=on_change, inputs=[checkbox], outputs=[checkbox])

        return checkbox

    def _build_status_text(self, extension, control: Dict[str, Any]) -> gr.Textbox:
        """Build status text display component."""
        label = control.get("label", "Status")
        default = control.get("default", "")

        textbox = gr.Textbox(
            value=default,
            label=label,
            interactive=False
        )

        # Note: Updating this requires polling setup in main app
        # We'll handle that in control_center.py

        return textbox

    def _build_section_header(self, control: Dict[str, Any]) -> gr.Markdown:
        """Build section header."""
        label = control.get("label", "Section")
        return gr.Markdown(f"### {label}")

    def _build_text_area(self, extension, control: Dict[str, Any]) -> gr.Textbox:
        """Build text area component."""
        label = control.get("label", "Text")
        placeholder = control.get("placeholder", "")
        rows = control.get("rows", 4)
        endpoint = control.get("endpoint", "")
        method = control.get("method", "POST")

        textbox = gr.Textbox(
            label=label,
            placeholder=placeholder,
            lines=rows
        )

        # Wire up change handler
        if endpoint:
            def on_change(value):
                try:
                    url = f"{extension.api_base_url}{endpoint}"
                    payload = {"text": value}
                    response = requests.request(method, url, json=payload, timeout=2)
                    return value
                except Exception as e:
                    logger.error(f"Text area change error: {e}")
                    return value

            textbox.change(fn=on_change, inputs=[textbox], outputs=[textbox])

        return textbox
