# modify_motor_pid.py
"""
Reachy Mini Motor PID Configuration Tool

A utility for reading and modifying PID (Proportional-Integral-Derivative) gains
on Reachy Mini robot motors. Supports auto-detection of serial ports and includes
validation to ensure sane PID values.

Author: Daniel Ritchie
GitHub: @brainwavecoder9
Discord: LeDaniel (quantumpoet)

Usage:
    python modify_motor_pid.py                      # Print current values Only (No Changes)
    python modify_motor_pid.py -d 200               # Set D gain
    python modify_motor_pid.py -p 400 -i 0 -d 200   # Set multiple gains
    python modify_motor_pid.py --motor-id 11 -d 150 # Configure different motor
    python modify_motor_pid.py --serialport COM6    # Specify serial port
    python modify_motor_pid.py --help               # Show detailed help

For more detailed parameter information, run with --help flag.
"""

from reachy_mini_motor_controller import ReachyMiniPyControlLoop
from reachy_mini.daemon.utils import find_serial_port
from datetime import timedelta
import struct
import time
import argparse

__version__ = "1.0.0"
__author__ = "Daniel Ritchie"
__github__ = "@brainwavecoder9"

# PID value ranges (based on typical motor controller limits)
PID_RANGES = {
    'p': (0, 16383),      # P gain: 0 to 16383 (typical max for 14-bit)
    'i': (0, 16383),      # I gain: 0 to 16383
    'd': (0, 16383),      # D gain: 0 to 16383
}

# Register addresses for PID gains
PID_REGISTERS = {
    'p': 84,  # P gain register (Kpp)
    'i': 82,  # I gain register (Kpi)
    'd': 80,  # D gain register (Kpd)
}

# Conversion factors for display
PID_CONVERSION = {
    'p': 128,   # Kpp = P / 128
    'i': 2048,  # Kpi = I / 2048
    'd': 16,    # Kpd = D / 16
}

def validate_pid_value(gain_type, value):
    """Validate that a PID value is within acceptable range."""
    min_val, max_val = PID_RANGES[gain_type]
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{gain_type.upper()} gain must be between {min_val} and {max_val}, got {value}"
        )
    return value

def read_pid_values(controller, motor_id):
    """Read current P, I, D values from the motor."""
    p_val = struct.unpack('<H', bytes(controller.async_read_raw_bytes(motor_id, PID_REGISTERS['p'], 2)))[0]
    i_val = struct.unpack('<H', bytes(controller.async_read_raw_bytes(motor_id, PID_REGISTERS['i'], 2)))[0]
    d_val = struct.unpack('<H', bytes(controller.async_read_raw_bytes(motor_id, PID_REGISTERS['d'], 2)))[0]
    return p_val, i_val, d_val

def print_pid_values(p, i, d, prefix=""):
    """Pretty print PID values."""
    print(f"{prefix}P = {p} (Kpp = {p/PID_CONVERSION['p']:.4f})")
    print(f"{prefix}I = {i} (Kpi = {i/PID_CONVERSION['i']:.6f})")
    print(f"{prefix}D = {d} (Kpd = {d/PID_CONVERSION['d']:.4f})")

def main():
    parser = argparse.ArgumentParser(
        description="""
Reachy Mini Motor PID Configuration Tool

Read and modify PID gains for Reachy Mini motor controllers. This tool provides
safe, validated access to motor PID parameters with automatic serial port detection
and clear before/after reporting.

PID Parameters:
  P (Proportional): Determines response to current error (typical: 400)
  I (Integral):     Determines response to accumulated error (typical: 0)
  D (Derivative):   Determines response to rate of error change (typical: 0-200)

Register Addresses:
  P Gain: Address 84 (Kpp = P / 128)
  I Gain: Address 82 (Kpi = I / 2048)
  D Gain: Address 80 (Kpd = D / 16)

Common Motor IDs:
  10: body_yaw (base rotation)
  11-13: Other body motors
  
Author: Daniel Ritchie (@brainwavecoder9)
Discord: LeDaniel (quantumpoet)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read current PID values without making changes
  python modify_motor_pid.py
  
  # Add damping by setting D gain (helps reduce oscillation)
  python modify_motor_pid.py -d 200
  
  # Set multiple gains at once
  python modify_motor_pid.py -p 400 -d 200
  
  # Configure all PID parameters
  python modify_motor_pid.py -p 400 -i 0 -d 200
  
  # Configure a different motor (e.g., motor ID 11)
  python modify_motor_pid.py --motor-id 11 -d 150
  
  # Manually specify serial port
  python modify_motor_pid.py --serialport COM6 -d 200
  
  # Remove D gain (set back to zero)
  python modify_motor_pid.py -d 0

Safety Notes:
  - Values are validated before being written to motors
  - Torque is automatically disabled during changes
  - Changes are verified after writing
  - Torque is re-enabled after successful changes

For issues or questions:
  GitHub: https://github.com/brainwavecollective/reachy_mini
  Discord: LeDaniel (quantumpoet)
        """
    )
    
    parser.add_argument('-p', '--p-gain', type=int, metavar='N',
                        help=f'P gain value (range: {PID_RANGES["p"][0]}-{PID_RANGES["p"][1]})')
    parser.add_argument('-i', '--i-gain', type=int, metavar='N',
                        help=f'I gain value (range: {PID_RANGES["i"][0]}-{PID_RANGES["i"][1]})')
    parser.add_argument('-d', '--d-gain', type=int, metavar='N',
                        help=f'D gain value (range: {PID_RANGES["d"][0]}-{PID_RANGES["d"][1]})')
    parser.add_argument('--motor-id', type=int, default=10,
                        help='Motor ID to configure (default: 10 for body_yaw)')
    parser.add_argument('--serialport', type=str, default='auto',
                        help='Serial port (default: auto-detect)')
    parser.add_argument('--version', action='version', 
                        version=f'%(prog)s {__version__} by {__author__}')
    
    args = parser.parse_args()
    
    # Validate PID values if provided
    changes = {}
    if args.p_gain is not None:
        changes['p'] = validate_pid_value('p', args.p_gain)
    if args.i_gain is not None:
        changes['i'] = validate_pid_value('i', args.i_gain)
    if args.d_gain is not None:
        changes['d'] = validate_pid_value('d', args.d_gain)
    
    # Auto-detect serial port
    serialport = args.serialport
    if serialport == "auto":
        print("Auto-detecting Reachy Mini serial port...")
        ports = find_serial_port(wireless_version=False)
        
        if len(ports) == 0:
            raise RuntimeError(
                "No Reachy Mini serial port found. "
                "Check USB connection and permissions. "
                "Or specify port with --serialport COM6"
            )
        elif len(ports) > 1:
            raise RuntimeError(
                f"Multiple Reachy Mini serial ports found: {ports}. "
                f"Please specify port with --serialport COM6"
            )
        
        serialport = ports[0]
        print(f"✅ Found Reachy Mini serial port: {serialport}\n")
    
    motor_id = args.motor_id
    print(f"Connecting to motor ID {motor_id}...")
    
    c = ReachyMiniPyControlLoop(
        serialport,
        read_position_loop_period=timedelta(seconds=0.02),
        allowed_retries=5,
        stats_pub_period=None,
    )
    
    try:
        # Read current values
        print("\n" + "="*50)
        print("CURRENT PID VALUES:")
        print("="*50)
        p_before, i_before, d_before = read_pid_values(c, motor_id)
        print_pid_values(p_before, i_before, d_before)
        
        # If no changes requested, just exit
        if not changes:
            print("\n✅ No changes requested. Done!")
            return
        
        # Apply changes
        print("\n" + "="*50)
        print("APPLYING CHANGES:")
        print("="*50)
        
        c.disable_torque()
        time.sleep(0.2)
        
        for gain_type, new_value in changes.items():
            register = PID_REGISTERS[gain_type]
            print(f"Setting {gain_type.upper()} = {new_value}...")
            c.async_write_raw_bytes(motor_id, register, list(struct.pack('<H', new_value)))
            time.sleep(0.1)
        
        # Verify changes
        print("\n" + "="*50)
        print("VERIFYING NEW VALUES:")
        print("="*50)
        p_after, i_after, d_after = read_pid_values(c, motor_id)
        print_pid_values(p_after, i_after, d_after)
        
        # Show what changed
        print("\n" + "="*50)
        print("SUMMARY OF CHANGES:")
        print("="*50)
        if p_before != p_after:
            print(f"P: {p_before} → {p_after} (Kpp: {p_before/PID_CONVERSION['p']:.4f} → {p_after/PID_CONVERSION['p']:.4f})")
        if i_before != i_after:
            print(f"I: {i_before} → {i_after} (Kpi: {i_before/PID_CONVERSION['i']:.6f} → {i_after/PID_CONVERSION['i']:.6f})")
        if d_before != d_after:
            print(f"D: {d_before} → {d_after} (Kpd: {d_before/PID_CONVERSION['d']:.4f} → {d_after/PID_CONVERSION['d']:.4f})")
        
        c.enable_torque()
        print("\n✅ Done! Torque re-enabled. Test the motor now.")
        
    finally:
        c.close()

if __name__ == "__main__":
    main()
