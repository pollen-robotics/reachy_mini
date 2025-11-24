# Changelog of Reachy Mini Audio Firmware


## 2.1.3

*For Beta units only*

Fixes the initialization issue on Reachy Mini beta hardware. An additional 2-second delay is added during initialization to prevent the XMOS chip from starting before the other components.

There is no need to apply this firmware to the Lite and Wireless versions, as the issue is fixed at the hardware level.

## 2.1.2

Improved parameters for acoustic echo cancellation (AEC).
PP_DTSENSITIVE is set to 1 by default.