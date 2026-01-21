# Reachy Mini Lite Hardware Datasheet

## Global description

![Dimensions](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/reachy_mini_dimensions.png)

- Dimensions: 30x20x15.5cm (extended)
- Mass: 1.350 Kg
- Material: ABS, PC, Aluminium, Steel
- Dofs:
    - Head : 6 dofs  (3 rotations & 3 translations)
    - Body : 1 rotation
    - Antennas : 1 rotation (x2)
    
    ![Dofs](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/degrees_of_freedom.png)
    
    ![dof_table](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/dof_table.png)

    
- Power input voltage: 6.8 - 7.6V
- Wide angle camera:  120° - 12MP - Autofocus
- Sound : Mic Array + Speaker

![Mics_and_camera](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/mic_and_camera.png)


![back_interface](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/back_interface.png)

- Control: USB-C Connection for computer interface. 
Please note that it will *not* charge through this USB port.
Reachy Mini Lite is a device plugged on a computer

![Components](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/reachy_mini_components.png)


## Specific components

- Motors specs
    - Base: 1x Custom Dynamixel XC330-M288-PG ([XC330-M288-T](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/) with plastic gear)
    - Antennas: 2x Dynamixel [XL330-M077-T](https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/)
    - Stewart Platform: 6x Dynamixel [XL330-M288-T](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
    
    ![Motors](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/motors_detail.png)
    
- Mic Array Board
    - 4 PDM MEMS digital mics
    - 16 kHz max sample rate / -26 dB FS Sensitivity / 64 dBA SNR
    - Based on Seeed Studio’s reSpeaker XMOS XVF3800
- Camera
    - Raspberry pi camera v3 wide angle
        - Sony IMX708
        - 12MP
        - Auto focus
        - I2C*~1 x MIDI DSI connection
- 5W @4Ohms Speaker
- Power board
    - Input voltage: 6.8 - 7.6V
    
    ![Electronics](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/electronics.png)
    
    
- Lite controller board:
    - 6.8 - 7.6 V supplied from the Power Board
    - Dynamixel Motor TTL connection
    - Camera CSI connection
    - Mics array connection
    - USB-C input, used for the bottom interface's usb port. Please note that it will *not* charge through this USB port. Reachy Mini Lite is a device plugged on a computer