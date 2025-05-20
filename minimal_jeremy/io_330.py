import pypot.dynamixel.conversion as conv
from pypot.dynamixel.io.abstract_io import AbstractDxlIO, _DxlAccess, _DxlControl
from pypot.dynamixel.protocol import v2 as v2

max_pos = 4096
max_deg = 360
max_current = 1750


def dxl_to_degree(value, model):
    return round(((max_deg * float(value)) / (max_pos - 1)) - (max_deg / 2), 2)


def dxl_to_current(value, model):
    if value > 0x7FFF:
        value = value - 65536
    return value


def current_to_dxl(value, model):
    if value < 0:
        value = value + 65536
    return int(value)


def degree_to_dxl(value, model):
    pos = int(round((max_pos - 1) * ((max_deg / 2 + float(value)) / max_deg), 0))
    pos = min(max(pos, 0), max_pos - 1)

    return pos


def dxl_to_velocity(value, model):
    if value > 2 ** (4 * 8 - 1):
        value = value - 2 ** (4 * 8)
    return value * 0.229


class Dxl330IO(AbstractDxlIO):
    _protocol = v2


controls = {
    # EEPROM
    "model": {
        "address": 0x00,
        "access": _DxlAccess.readonly,
        "dxl_to_si": conv.dxl_to_model,
    },
    "firmware": {"address": 0x026, "length": 1, "access": _DxlAccess.readonly},
    "id": {
        "address": 0x07,
        "length": 1,
        "access": _DxlAccess.writeonly,
        "setter_name": "change_id",
    },
    "baudrate": {
        "address": 0x08,
        "length": 1,
        "access": _DxlAccess.writeonly,
        "setter_name": "change_baudrate",
        "si_to_dxl": conv.baudrate_to_dxl,
    },
    "return delay time": {
        "address": 0x09,
        "length": 1,
        "dxl_to_si": conv.dxl_to_rdt,
        "si_to_dxl": conv.rdt_to_dxl,
    },
    "angle limit": {
        "address": 0x30,
        "nb_elem": 2,
        "dxl_to_si": lambda value, model: (
            dxl_to_degree(value[0], model),
            dxl_to_degree(value[1], model),
        ),
        "si_to_dxl": lambda value, model: (
            degree_to_dxl(value[0], model),
            degree_to_dxl(value[1], model),
        ),
    },
    "control mode": {
        "address": 0x10,
        "length": 1,
        "dxl_to_si": conv.dxl_to_control_mode,
        "si_to_dxl": conv.control_mode_to_dxl,
    },
    "operating mode": {
        "address": 0xB,
        "length": 1,
    },
    "highest temperature limit": {
        "address": 0x1F,
        "length": 1,
        "dxl_to_si": conv.dxl_to_temperature,
        "si_to_dxl": conv.temperature_to_dxl,
    },
    "voltage limit": {
        "address": 0x20,
        "length": 1,
        "nb_elem": 2,
        "dxl_to_si": lambda value, model: (
            conv.dxl_to_voltage(value[0], model),
            conv.dxl_to_voltage(value[1], model),
        ),
        "si_to_dxl": lambda value, model: (
            conv.voltage_to_dxl(value[0], model),
            conv.voltage_to_dxl(value[1], model),
        ),
    },
    "current limit": {
        "address": 0x26,
        "length": 2,
    },
    # RAM
    "torque_enable": {
        "address": 0x40,
        "length": 1,
        "dxl_to_si": conv.dxl_to_bool,
        "si_to_dxl": conv.bool_to_dxl,
        "getter_name": "is_torque_enabled",
        "setter_name": "_set_torque_enable",
    },
    "LED": {
        "address": 0x41,
        "length": 1,
        "dxl_to_si": conv.dxl_to_bool,
        "si_to_dxl": conv.bool_to_dxl,
        "setter_name": "_set_LED",
        "getter_name": "is_led_on",
    },
    "LED color": {
        "address": 0x19,
        "length": 1,
        "dxl_to_si": conv.dxl_to_led_color,
        "si_to_dxl": conv.led_color_to_dxl,
    },
    "pid gain": {
        "address": 0x50,
        "length": 2,
        "nb_elem": 3,
        # "dxl_to_si": conv.dxl_to_pid,
        # "si_to_dxl": conv.pid_to_dxl,
    },
    "goal position": {
        "address": 0x74,
        "length": 4,
        "dxl_to_si": dxl_to_degree,
        "si_to_dxl": degree_to_dxl,
    },
    "present velocity": {
        "address": 0x80,
        "length": 4,
        "access": _DxlAccess.readonly,
        "dxl_to_si": dxl_to_velocity,
    },
    "present position": {
        "address": 0x84,
        "length": 4,
        "access": _DxlAccess.readonly,
        "dxl_to_si": dxl_to_degree,
    },
    "present current": {
        "address": 0x7E,
        "length": 2,
        "access": _DxlAccess.readonly,
        "dxl_to_si": dxl_to_current,
    },
    "goal current": {
        "address": 0x66,
        "length": 2,
        "dxl_to_si": dxl_to_current,
        "si_to_dxl": current_to_dxl,
    },
    "present input voltage": {
        "address": 0x90,
        "length": 2,
        "access": _DxlAccess.readonly,
        "dxl_to_si": conv.dxl_to_voltage,
    },
    "present temperature": {
        "address": 0x92,
        "length": 1,
        "access": _DxlAccess.readonly,
        "dxl_to_si": conv.dxl_to_temperature,
    },
}


def _add_control(
    name,
    address,
    length=2,
    nb_elem=1,
    access=_DxlAccess.readwrite,
    models=[
        "XL-330",
    ],
    dxl_to_si=lambda val, model: val,
    si_to_dxl=lambda val, model: val,
    getter_name=None,
    setter_name=None,
):
    control = _DxlControl(
        name,
        address,
        length,
        nb_elem,
        access,
        models,
        dxl_to_si,
        si_to_dxl,
        getter_name,
        setter_name,
    )

    Dxl330IO._generate_accessors(control)


for name, args in controls.items():
    _add_control(name, **args)
