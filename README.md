# Reachy Mini

[![Ask on HuggingChat](https://img.shields.io/badge/Ask_on-HuggingChat-yellow?logo=huggingface&logoColor=yellow&style=for-the-badge)](https://huggingface.co/chat/?attachments=https%3A%2F%2Fgist.githubusercontent.com%2FFabienDanieau%2F919e1d7468fb16e70dbe984bdc277bba%2Fraw%2Fdoc_reachy_mini_full.md&prompt=Read%20this%20documentation%20about%20Reachy%20Mini%20so%20I%20can%20ask%20questions%20about%20it.)

<!-- > ⚠️ Reachy Mini is still in beta. Expect bugs, some of them we won't fix right away if they are not a priority. -->

[Reachy Mini](https://huggingface.co/blog/reachy-mini) is a cute, expressive, open source robot designed to make embodied AI tangible and fun. It is built primarily for AI builders who want to move beyond screens and experiment with models, agents, and interactions in the physical world. Affordable, easy to use, and hackable, Reachy Mini lets you focus on building and iterating on AI behaviors rather than dealing with complex robotics setups.

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

Reachy Mini fits naturally into modern AI workflows. You can connect it to vision, speech, and language models, run agentic pipelines, and iterate quickly using familiar tools from the Hugging Face ecosystem.

### Start building without the robot

You do not need a physical Reachy Mini to get started. A simulated version is available in [MuJoCo](https://mujoco.org), allowing you to prototype, test, and debug applications before deploying them on real hardware. The simulation behaves like the Lite version, with the full software stack running on your computer.

### Hardware

Reachy Mini hardware comes in two flavors.

#### Reachy Mini
The full, autonomous experience. It features onboard computing to run apps and interactions directly on the robot, enabling smoother and more immersive embodied AI experiments. You can connect to it wirelessly or by Ethernet, from any device including a laptop, phone, or tablet.

It includes a built in battery providing around 1.5 to 2 hours of operation, and it can also run while plugged into a power outlet.

*Note: Some more compute intensive applications may require running part of the processing on a remote computer or server.*

#### Reachy Mini Lite
A more minimal version designed for learning, tinkering, and development. It requires a separate computer to run applications and connects via USB. A dedicated desktop application makes it easy to connect to the robot and launch apps.

It has no battery and only works when plugged into a power outlet. Reachy Mini Lite can run the same applications as Reachy Mini, except those that rely on the accelerometer, which is only available on Reachy Mini.

**Learn more and buy Reachy Mini at https://huggingface.co/blog/reachy-mini**

## Getting started

Reachy Mini robots are sold as kits and need to be assembled. Assembly typically takes 2 to 3 hours for a first time build and is guided step by step by a detailed assembly guide included in the box.

Detailed instructions are available in the documentation:

- [Assembly + first boot](docs/getting_started.md)
- [Software](docs/software/readme.md)
- [Simulation](docs/simulation.md)
- [API & SDK](docs/api.md)
- [Examples & apps](docs/examples.md)
- [Troubleshooting](docs/troubleshooting.md)
- [3D Models](TODO)

## Join the community

Reachy Mini is built as an open source project and grows with its community. If you have questions, want to share what you are building, get help, or contribute ideas, join the Reachy community on Discord.

👉 https://discord.gg/2bAhWfXme9


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

The robot design files are licensed under the [Creative Commons BY-SA-NC](TODO) license.

## Open source & contribution

This project is actively developed and maintained by the [Pollen Robotics team](https://www.pollen-robotics.com) and the [Hugging Face team](https://huggingface.co/).

Contributions are welcome. Feel free to open issues to report bugs or request features, and submit pull requests if you want to contribute code.

