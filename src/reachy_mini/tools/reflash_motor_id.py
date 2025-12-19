import argparse
from importlib.resources import files

import questionary
from rich.console import Console
from rustypot import Xl330PyController

import reachy_mini
from reachy_mini.daemon.utils import find_serial_port
from reachy_mini.utils.hardware_config.parser import parse_yaml_config

parser = argparse.ArgumentParser(
    description="Reflash Reachy Mini motors' firmware.",
)
parser.add_argument(
    "--serialport",
    type=str,
    required=False,
    default=None,
    help="Serial port of the Reachy Mini (e.g. /dev/ttyUSB0 or COM3). "
    "If not specified, the script will try to automatically find it.",
)
args = parser.parse_args()

console = Console()

config_file_path = str(
    files(reachy_mini).joinpath("assets/config/hardware_config.yaml")
)
config = parse_yaml_config(config_file_path)
motors = config.motors
name_to_id = {m: config.motors[m].id for m in config.motors}
id_to_name = {v: k for k, v in name_to_id.items()}

console.print("Reachy Mini - Reflash Motor ID Tool\n", style="bold green")
console.print(
    "[Warning] : Make sure that only one motor is connected before proceeding ! .\n",
    style="bold red",
)
if args.serialport is None:
    console.print(
        "Which version of Reachy Mini are you using?",
    )
    wireless_choice = questionary.select(
        ">",
        [
            questionary.Choice("Lite", value=False),
            questionary.Choice("Wireless", value=True),
        ],
    ).ask()
    ports = find_serial_port(wireless_version=wireless_choice)

    if len(ports) == 0:
        raise RuntimeError(
            "No Reachy Mini serial port found. "
            "Check USB connection and permissions. "
            "Or directly specify the serial port using --serialport."
        )
    elif len(ports) > 1:
        raise RuntimeError(
            f"Multiple Reachy Mini serial ports found {ports}."
            "Please specify the serial port using --serialport."
        )

    serialport = ports[0]
    console.print(f"Found Reachy Mini serial port: {serialport}", style="green")
else:
    serialport = args.serialport

c = Xl330PyController(serialport, baudrate=1000000, timeout=0.01)

found_ids = []
for i in range(254):
    ret = c.ping(i)
    if ret:
        found_ids.append(i)

if len(found_ids) == 0:
    console.print("No motor found. Please check the connection.", style="bold red")
    exit(1)
if len(found_ids) > 1:
    console.print(
        f"Multiple motors found with IDs: {found_ids}. Please make sure only one motor is connected.",
        style="bold red",
    )
    exit(1)

current_id = found_ids[0]
console.print(f"Found motor with ID: {current_id}", style="green")

new_id = questionary.select(
    "What motor are you reflashing ? :",
    choices=[
        questionary.Choice(motor_name, value=name_to_id[motor_name])
        for motor_name in motors
    ],
).ask()


new_id = int(new_id)
if new_id == current_id:
    console.print(
        "The new ID is the same as the current ID. No changes made.", style="yellow"
    )
    exit(0)

c.write_id(current_id, new_id)

console.print("✅ Motor ID updated successfully", style="bold green")
console.print(f"flashed {id_to_name[new_id]} with ID {new_id}", style="green")
