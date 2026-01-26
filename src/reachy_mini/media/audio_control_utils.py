"""Allows tuning of the XMOS XVF3800 chip integrated in the Reachy Mini Audio card.

Example usage:

    # Read a parameter
    python audio_control_utils.py AUDIO_MGR_OP_L
    # Output:
    # ReadCMD: cmdid: 143, resid: 35, response: array('B', [0, 8, 0])
    # AUDIO_MGR_OP_L: [0, 8, 0]

    # Write a parameter
    python audio_control_utils.py AUDIO_MGR_OP_L --values 3 0
    # Output:
    # Writing to AUDIO_MGR_OP_L with values: [3, 0]
    # WriteCMD: cmdid: 15, resid: 35, payload: [3, 0]
    # Write operation completed successfully

More details about the parameters is available at:
https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/user_guide/AA_control_command_appendix.html
"""

import argparse
import logging
import struct
import sys
import time
from typing import Any, Optional

import usb.core
import usb.util
from libusb_package import get_libusb1_backend

CONTROL_SUCCESS = 0
SERVICER_COMMAND_RETRY = 64

# name, resid, cmdid, length, type
PARAMETERS = {
    # APPLICATION_SERVICER_RESID commands
    "VERSION": (48, 0, 3, "ro", "uint8"),
    "BLD_MSG": (48, 1, 50, "ro", "char"),
    "BLD_HOST": (48, 2, 30, "ro", "char"),
    "BLD_REPO_HASH": (48, 3, 40, "ro", "char"),
    "BLD_MODIFIED": (48, 4, 6, "ro", "char"),
    "BOOT_STATUS": (48, 5, 3, "ro", "char"),
    "TEST_CORE_BURN": (48, 6, 1, "rw", "uint8"),
    "REBOOT": (48, 7, 1, "wo", "uint8"),
    "USB_BIT_DEPTH": (48, 8, 2, "rw", "uint8"),
    "SAVE_CONFIGURATION": (48, 9, 1, "wo", "uint8"),
    "CLEAR_CONFIGURATION": (48, 10, 1, "wo", "uint8"),
    # AEC_RESID commands
    "SHF_BYPASS": (33, 70, 1, "rw", "uint8"),
    "AEC_NUM_MICS": (33, 71, 1, "ro", "int32"),
    "AEC_NUM_FARENDS": (33, 72, 1, "ro", "int32"),
    "AEC_MIC_ARRAY_TYPE": (33, 73, 1, "ro", "int32"),
    "AEC_MIC_ARRAY_GEO": (33, 74, 12, "ro", "float"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians"),
    "TEST_AEC_DISABLE_CONTROL": (33, 76, 1, "wo", "uint32"),
    "AEC_CURRENT_IDLE_TIME": (33, 77, 1, "ro", "uint32"),
    "AEC_MIN_IDLE_TIME": (33, 78, 1, "ro", "uint32"),
    "AEC_RESET_MIN_IDLE_TIME": (33, 79, 1, "wo", "uint32"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8"),
    "SPECIAL_CMD_AEC_FAR_MIC_INDEX": (33, 90, 2, "wo", "int32"),
    "SPECIAL_CMD_AEC_FILTER_COEFF_START_OFFSET": (33, 91, 1, "rw", "int32"),
    "SPECIAL_CMD_AEC_FILTER_COEFFS": (33, 92, 15, "rw", "float"),
    "SPECIAL_CMD_AEC_FILTER_LENGTH": (33, 93, 1, "ro", "int32"),
    "AEC_FILTER_CMD_ABORT": (33, 94, 1, "wo", "int32"),
    "AEC_AECPATHCHANGE": (33, 0, 1, "ro", "int32"),
    "AEC_HPFONOFF": (33, 1, 1, "rw", "int32"),
    "AEC_AECSILENCELEVEL": (33, 2, 2, "rw", "float"),
    "AEC_AECCONVERGED": (33, 3, 1, "ro", "int32"),
    "AEC_AECEMPHASISONOFF": (33, 4, 1, "rw", "int32"),
    "AEC_FAR_EXTGAIN": (33, 5, 1, "rw", "float"),
    "AEC_PCD_COUPLINGI": (33, 6, 1, "rw", "float"),
    "AEC_PCD_MINTHR": (33, 7, 1, "rw", "float"),
    "AEC_PCD_MAXTHR": (33, 8, 1, "rw", "float"),
    "AEC_RT60": (33, 9, 1, "ro", "float"),
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32"),
    "AEC_ASROUTGAIN": (33, 36, 1, "rw", "float"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32"),
    "AEC_FIXEDBEAMNOISETHR": (33, 38, 2, "rw", "float"),
    # AUDIO_MGR_RESID commands
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float"),
    "AUDIO_MGR_REF_GAIN": (35, 1, 1, "rw", "float"),
    "AUDIO_MGR_CURRENT_IDLE_TIME": (35, 2, 1, "ro", "int32"),
    "AUDIO_MGR_MIN_IDLE_TIME": (35, 3, 1, "ro", "int32"),
    "AUDIO_MGR_RESET_MIN_IDLE_TIME": (35, 4, 1, "wo", "int32"),
    "MAX_CONTROL_TIME": (35, 5, 1, "ro", "int32"),
    "RESET_MAX_CONTROL_TIME": (35, 6, 1, "wo", "int32"),
    "I2S_CURRENT_IDLE_TIME": (35, 7, 1, "ro", "int32"),
    "I2S_MIN_IDLE_TIME": (35, 8, 1, "ro", "int32"),
    "I2S_RESET_MIN_IDLE_TIME": (35, 9, 1, "wo", "int32"),
    "I2S_INPUT_PACKED": (35, 10, 1, "rw", "uint8"),
    "AUDIO_MGR_SELECTED_AZIMUTHS": (35, 11, 2, "ro", "radians"),
    "AUDIO_MGR_SELECTED_CHANNELS": (35, 12, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_PACKED": (35, 13, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_UPSAMPLE": (35, 14, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_L_PK0": (35, 16, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_L_PK1": (35, 17, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_L_PK2": (35, 18, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R_PK0": (35, 20, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R_PK1": (35, 21, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R_PK2": (35, 22, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_ALL": (35, 23, 12, "rw", "uint8"),
    "I2S_INACTIVE": (35, 24, 1, "ro", "uint8"),
    "AUDIO_MGR_FAR_END_DSP_ENABLE": (35, 25, 1, "rw", "uint8"),
    "AUDIO_MGR_SYS_DELAY": (35, 26, 1, "rw", "int32"),
    "I2S_DAC_DSP_ENABLE": (35, 27, 1, "rw", "uint8"),
    # GPO_SERVICER_RESID commands
    "GPO_READ_VALUES": (20, 0, 5, "ro", "uint8"),
    "GPO_WRITE_VALUE": (20, 1, 2, "wo", "uint8"),
    "GPO_PORT_PIN_INDEX": (20, 2, 2, "rw", "uint32"),
    "GPO_PIN_VAL": (20, 3, 3, "wo", "uint8"),
    "GPO_PIN_ACTIVE_LEVEL": (20, 4, 1, "rw", "uint32"),
    "GPO_PIN_PWM_DUTY": (20, 5, 1, "rw", "uint8"),
    "GPO_PIN_FLASH_MASK": (20, 6, 1, "rw", "uint32"),
    "LED_EFFECT": (20, 12, 1, "rw", "uint8"),
    "LED_BRIGHTNESS": (20, 13, 1, "rw", "uint8"),
    "LED_GAMMIFY": (20, 14, 1, "rw", "uint8"),
    "LED_SPEED": (20, 15, 1, "rw", "uint8"),
    "LED_COLOR": (20, 16, 1, "rw", "uint32"),
    "LED_DOA_COLOR": (20, 17, 2, "rw", "uint32"),
    "DOA_VALUE": (20, 18, 2, "ro", "uint32"),
    "DOA_VALUE_RADIANS": (20, 19, 2, "ro", "radians"),
    # PP_RESID commands
    "PP_CURRENT_IDLE_TIME": (17, 70, 1, "ro", "uint32"),
    "PP_MIN_IDLE_TIME": (17, 71, 1, "ro", "uint32"),
    "PP_RESET_MIN_IDLE_TIME": (17, 72, 1, "wo", "uint32"),
    "SPECIAL_CMD_PP_NLMODEL_NROW_NCOL": (17, 90, 2, "ro", "int32"),
    "SPECIAL_CMD_NLMODEL_START": (17, 91, 1, "wo", "int32"),
    "SPECIAL_CMD_NLMODEL_COEFF_START_OFFSET": (17, 92, 1, "rw", "int32"),
    "SPECIAL_CMD_PP_NLMODEL": (17, 93, 15, "rw", "float"),
    "PP_NL_MODEL_CMD_ABORT": (17, 94, 1, "wo", "int32"),
    "SPECIAL_CMD_PP_NLMODEL_BAND": (17, 95, 1, "rw", "uint8"),
    "SPECIAL_CMD_PP_EQUALIZATION_NUM_BANDS": (17, 96, 1, "ro", "int32"),
    "SPECIAL_CMD_EQUALIZATION_START": (17, 97, 1, "wo", "int32"),
    "SPECIAL_CMD_EQUALIZATION_COEFF_START_OFFSET": (17, 98, 1, "rw", "int32"),
    "SPECIAL_CMD_PP_EQUALIZATION": (17, 99, 15, "rw", "float"),
    "PP_EQUALIZATION_CMD_ABORT": (17, 100, 1, "wo", "int32"),
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32"),
    "PP_AGCMAXGAIN": (17, 11, 1, "rw", "float"),
    "PP_AGCDESIREDLEVEL": (17, 12, 1, "rw", "float"),
    "PP_AGCGAIN": (17, 13, 1, "rw", "float"),
    "PP_AGCTIME": (17, 14, 1, "rw", "float"),
    "PP_AGCFASTTIME": (17, 15, 1, "rw", "float"),
    "PP_AGCALPHAFASTGAIN": (17, 16, 1, "rw", "float"),
    "PP_AGCALPHASLOW": (17, 17, 1, "rw", "float"),
    "PP_AGCALPHAFAST": (17, 18, 1, "rw", "float"),
    "PP_LIMITONOFF": (17, 19, 1, "rw", "int32"),
    "PP_LIMITPLIMIT": (17, 20, 1, "rw", "float"),
    "PP_MIN_NS": (17, 21, 1, "rw", "float"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float"),
    "PP_ECHOONOFF": (17, 23, 1, "rw", "int32"),
    "PP_GAMMA_E": (17, 24, 1, "rw", "float"),
    "PP_GAMMA_ETAIL": (17, 25, 1, "rw", "float"),
    "PP_GAMMA_ENL": (17, 26, 1, "rw", "float"),
    "PP_NLATTENONOFF": (17, 27, 1, "rw", "int32"),
    "PP_NLAEC_MODE": (17, 28, 1, "rw", "int32"),
    "PP_MGSCALE": (17, 29, 3, "rw", "float"),
    "PP_FMIN_SPEINDEX": (17, 30, 1, "rw", "float"),
    "PP_DTSENSITIVE": (17, 31, 1, "rw", "int32"),
    "PP_ATTNS_MODE": (17, 32, 1, "rw", "int32"),
    "PP_ATTNS_NOMINAL": (17, 33, 1, "rw", "float"),
    "PP_ATTNS_SLOPE": (17, 34, 1, "rw", "float"),
}


class ReSpeaker:
    """Class to interface with the ReSpeaker XVF3800 USB device."""

    TIMEOUT = 100000

    def __init__(self, dev: usb.core.Device) -> None:
        """Initialize the ReSpeaker interface with the given USB device."""
        self.dev = dev

    def write(self, name: str, data_list: Any) -> None:
        """Write data to a specified parameter on the ReSpeaker device."""
        try:
            data = PARAMETERS[name]
        except KeyError:
            return

        if data[3] == "ro":
            raise ValueError("{} is read-only".format(name))

        if len(data_list) != data[2]:
            raise ValueError("{} value count is not {}".format(name, data[2]))

        windex = data[0]  # resid
        wvalue = data[1]  # cmdid
        data_cnt = data[2]  # cnt
        data_type = data[4]  # data type
        payload = []  # type: ignore[var-annotated]

        if data_type == "float" or data_type == "radians":
            for i in range(data_cnt):
                payload += struct.pack(b"f", float(data_list[i]))
        elif data_type == "char":
            # For char arrays, convert string to bytes
            payload = (
                bytearray(data_list, "utf-8")  # type: ignore[assignment]
                if isinstance(data_list, str)
                else bytearray(data_list)
            )
        elif data_type == "uint8":
            for i in range(data_cnt):
                payload += data_list[i].to_bytes(1, byteorder="little")
        elif data_type == "uint32" or data_type == "int32":
            for i in range(data_cnt):
                payload += struct.pack(
                    b"I" if data_type == "uint32" else b"i", data_list[i]
                )
        else:
            # Default to int32 for other types
            for i in range(data_cnt):
                payload += struct.pack(b"i", data_list[i])

        logging.debug(
            "WriteCMD: cmdid: {}, resid: {}, payload: {}".format(
                wvalue, windex, payload
            )
        )

        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            wvalue,
            windex,
            payload,
            self.TIMEOUT,
        )

    def read(self, name: str) -> Any:
        """Read data from a specified parameter on the ReSpeaker device."""
        try:
            data = PARAMETERS[name]
        except KeyError:
            return

        read_attempts = 1
        windex = data[0]  # resid
        wvalue = 0x80 | data[1]  # cmdid
        data_cnt = data[2]  # cnt
        data_type = data[4]  # data type
        if data_type == "uint8" or data_type == "char":
            length = data_cnt + 1  # 1 byte for status
        elif (
            data_type == "float"
            or data_type == "radians"
            or data_type == "uint32"
            or data_type == "int32"
        ):
            length = data_cnt * 4 + 1  # 1 byte for status

        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN
            | usb.util.CTRL_TYPE_VENDOR
            | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            wvalue,
            windex,
            length,
            self.TIMEOUT,
        )
        while True:
            if read_attempts > 100:
                raise ValueError("Read attempt exceeds 100 times")
            if response[0] == CONTROL_SUCCESS:
                break
            elif response[0] == SERVICER_COMMAND_RETRY:
                read_attempts += 1
                response = self.dev.ctrl_transfer(
                    usb.util.CTRL_IN
                    | usb.util.CTRL_TYPE_VENDOR
                    | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,
                    wvalue,
                    windex,
                    length,
                    self.TIMEOUT,
                )
            else:
                raise ValueError("Unknown status code: {}".format(response[0]))
            time.sleep(0.01)

        logging.debug(
            "ReadCMD: cmdid: {}, resid: {}, response: {}".format(
                wvalue, windex, response
            )
        )

        if data_type == "uint8":
            result = response.tolist()
        elif data_type == "char":
            # For char arrays, convert bytes to string
            byte_data = response.tobytes()
            # Remove status byte and null terminators
            result = byte_data[1:].rstrip(b"\x00").decode("utf-8", errors="ignore")
        elif data_type == "radians" or data_type == "float":
            byte_data = response.tobytes()
            match_str = "<"
            for i in range(data_cnt):
                match_str += "f"
            result = struct.unpack(match_str, byte_data[1:])
        elif data_type == "uint32" or data_type == "int32":
            result = response.tolist()

        return result

    def close(self) -> None:
        """Close the interface."""
        usb.util.dispose_resources(self.dev)


def find(vid: int = 0x2886, pid: int = 0x001A) -> ReSpeaker | None:
    """Find and return the ReSpeaker USB device with the given Vendor ID and Product ID.

    Args:
        vid (int): USB Vendor ID to search for. Default: 0x2886 (XMOS).
        pid (int): USB Product ID to search for. Default: 0x001A (XMOS XVF3800).

    Returns:
        ReSpeaker | None: A ReSpeaker object if the device is found,
                         None otherwise.

    Note:
        This function searches for USB devices with the specified Vendor ID
        and Product ID using libusb backend. The default values target
        XMOS XVF3800 devices used in ReSpeaker microphone arrays.

    Example:
        ```python
        from reachy_mini.media.audio_control_utils import find

        # Find default ReSpeaker device
        respeaker = find()
        if respeaker is not None:
            print("Found ReSpeaker device")
            respeaker.close()

        # Find specific device
        custom_device = find(vid=0x1234, pid=0x5678)
        ```

    """
    dev = usb.core.find(idVendor=vid, idProduct=pid, backend=get_libusb1_backend())
    if not dev:
        return None

    return ReSpeaker(dev)


def init_respeaker_usb() -> Optional[ReSpeaker]:
    """Initialize the ReSpeaker USB device. Looks for both new and beta device IDs.

    Returns:
        Optional[ReSpeaker]: A ReSpeaker object if a compatible device is found,
                           None otherwise.

    Note:
        This function attempts to initialize a ReSpeaker microphone array by
        searching for USB devices with known Vendor and Product IDs. It tries:
        1. New Reachy Mini Audio firmware (0x38FB:0x1001) - preferred
        2. Old ReSpeaker firmware (0x2886:0x001A) - with warning to update

        The function handles USB backend errors gracefully and returns
        None if no compatible device is found or if initialization fails.

    Example:
        ```python
        from reachy_mini.media.audio_control_utils import init_respeaker_usb

        # Initialize ReSpeaker device
        respeaker = init_respeaker_usb()
        if respeaker is not None:
            print("ReSpeaker initialized successfully")
            # Use the device...
            doa = respeaker.read("DOA_VALUE_RADIANS")
            respeaker.close()
        else:
            print("No ReSpeaker device found")
        ```

    """
    try:
        # Try new firmware first
        dev = usb.core.find(
            idVendor=0x38FB, idProduct=0x1001, backend=get_libusb1_backend()
        )

        # If not found, try old firmware
        if dev is None:
            dev = usb.core.find(
                idVendor=0x2886, idProduct=0x001A, backend=get_libusb1_backend()
            )
            if dev is not None:
                logging.warning("Old firmware detected. Please update the firmware!")

        # If still not found, raise error
        if dev is None:
            logging.error("No Reachy Mini Audio USB device found!")
            return None

        return ReSpeaker(dev)

    except usb.core.NoBackendError:
        logging.error(
            "No USB backend was found! Make sure libusb_package is correctly installed with `pip install libusb_package`."
        )
        return None


def main() -> None:
    """Parse arguments and execute read/write commands."""
    parser = argparse.ArgumentParser(
        description="Reachy Mini Audio Host Control Script"
    )
    parser.add_argument(
        "command",
        choices=PARAMETERS.keys(),
        help="Command to execute (e.g., VERSION, DOA_VALUE, etc.)",
    )
    parser.add_argument(
        "--vid",
        type=lambda x: int(x, 0),
        default=0x38FB,
        help="Vendor ID (default: 0x38FB)",
    )
    parser.add_argument(
        "--pid",
        type=lambda x: int(x, 0),
        default=0x1001,
        help="Product ID (default: 0x1001)",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Values for write commands (only for write operations)",
    )

    args = parser.parse_args()

    # Allow user overrides if provided, else use known defaults
    if args.vid is not None and args.pid is not None:
        dev = find(vid=args.vid, pid=args.pid)
    else:
        dev = init_respeaker_usb()

    if not dev:
        print("No device found")
        sys.exit(1)

    try:
        if args.values:
            if PARAMETERS[args.command][3] == "ro":
                print(f"Error: {args.command} is read-only and cannot be written to")
                sys.exit(1)

            if (
                PARAMETERS[args.command][4] != "float"
                and PARAMETERS[args.command][4] != "radians"
            ):
                args.values = [int(v) for v in args.values]

            if PARAMETERS[args.command][2] != len(args.values):
                print(
                    f"Error: {args.command} value count is {PARAMETERS[args.command][2]}, but {len(args.values)} values provided"
                )
                sys.exit(1)

            print(f"Writing to {args.command} with values: {args.values}")
            dev.write(args.command, args.values)
            time.sleep(0.1)
            print("Write operation completed successfully")
        else:
            if PARAMETERS[args.command][3] == "wo":
                print(f"Error: {args.command} is write-only and cannot be read")
                sys.exit(1)

            result = dev.read(args.command)
            print(f"{args.command}: {result}")

    except Exception as e:
        error_msg = f"Error executing command {args.command}: {e}"
        print(error_msg)

        # Check if it's a permission error, so far only seen on Linux
        if (
            "Errno 13" in str(e)
            or "Access denied" in str(e)
            or "insufficient permissions" in str(e)
        ):
            print("\nThis looks like a permissions error.")
            print(
                "\n - You are most likely on Linux and need to adjust udev rules for USB permissions."
            )
            print(
                "\n - If you are not on Linux or have additional questions contact the team."
            )
        sys.exit(1)
    finally:
        dev.close()


if __name__ == "__main__":
    main()
