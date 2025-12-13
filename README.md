# Reachy Mini

[![Ask on HuggingChat](https://img.shields.io/badge/Ask_on-HuggingChat-yellow?logo=huggingface&logoColor=yellow&style=for-the-badge)](https://huggingface.co/chat/?attachments=https%3A%2F%2Fgist.githubusercontent.com%2FFabienDanieau%2F919e1d7468fb16e70dbe984bdc277bba%2Fraw%2Fdoc_reachy_mini_full.md&prompt=Read%20this%20documentation%20about%20Reachy%20Mini%20so%20I%20can%20ask%20questions%20about%20it.)

[Reachy Mini](https://huggingface.co/blog/reachy-mini) is an open source, expressive robot built for AI builders who want to experiment with models, agents, and interactions in the physical world.

Affordable, easy to use, and hackable, Reachy Mini lets you focus on building embodied AI behaviors rather than dealing with complex robotics setups.

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

## Getting started

Reachy Mini robots are sold as kits and need to be assembled. Assembly typically takes 2 to 3 hours for a first time build and is guided step by step by a detailed assembly guide included in the box.

You can start building with Reachy Mini even without the physical robot, thanks to simulation.

**Start here:**
- [Assembly + first boot](docs/getting_started.md)
- [Software setup](docs/software/README.md)
- [Simulation](docs/simulation.md)
- [API & SDK](docs/api.md)
- [Examples & apps](docs/examples.md)

## Hardware overview

Reachy Mini hardware comes in two versions.

### Reachy Mini
The full, autonomous experience. Applications run directly onboard, with WiFi or Ethernet connectivity. It includes a built in battery (around 1.5 to 2 hours of operation) and can also run while plugged into a power outlet.

👉 Learn more: [Reachy Mini hardware](docs/hardware/reachy-mini.md)

### Reachy Mini Lite
A more minimal version designed for learning, tinkering, and development. Applications run on a separate computer connected via USB. It has no battery and only works when plugged into a power outlet.

👉 Learn more: [Reachy Mini Lite hardware](docs/hardware/reachy-mini-lite.md)


## Build without hardware

You do not need a physical robot to get started. Reachy Mini is available as a simulated robot in [MuJoCo](https://mujoco.org), allowing you to prototype, test, and debug applications before deploying them on real hardware.

The simulation behaves like the Lite version, with the full software stack running on your computer.

## Software overview

Reachy Mini is designed to be simple to use, even for a first robotics project. Applications are managed by a daemon that handles the connection to the robot, whether it runs directly on the robot, on your computer, or in simulation.

Depending on your setup, you interact with Reachy Mini either through a web interface or a desktop application.

👉 Learn more: [Dashboard](docs/software/dashboard.md)

The same APIs and application logic work across Reachy Mini, Reachy Mini Lite, and simulation, making it easy to develop once and deploy anywhere.

To start building on Reachy Mini, you can install the Reachy Mini Python package:

```bash
pip install reachy-mini
```
👉 Learn more: [Software documentation](docs/software/README.md)


## Join the community

Reachy Mini is built as an open source project and grows with its community. Join the discussion, share what you are building, or get help on Discord.

👉 https://discord.gg/2bAhWfXme9


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

The robot design files are licensed under the [Creative Commons BY-SA-NC](TODO) license.


## Open source & contribution

This project is actively developed and maintained by the [Pollen Robotics team](https://www.pollen-robotics.com) and the [Hugging Face team](https://huggingface.co/).

Contributions are welcome. Feel free to open issues to report bugs or request features, and submit pull requests if you want to contribute code.
