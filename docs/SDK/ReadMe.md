# Build your own Reachy Mini app 🤖

> **You don't have to read the SDK docs.** Just describe what you want — your favourite AI coding agent does the rest.

Reachy Mini ships with a single file, [`AGENTS.md`](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md), that turns Claude Code, Cursor, Codex, Copilot, or any modern coding assistant into a Reachy Mini expert: SDK patterns, scaffolding playbooks, gotchas, the works.

---

## ⚡ The one-liner

Paste this into your agent, replacing the bracketed part with your idea:

> *I'd like to build a Reachy Mini app that **[hides when I say "boo"]**. Read https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md first — it has everything you need.*

That's it. The agent picks the right flavour, scaffolds the project, writes the code, and (for web apps) pushes it live to your Hugging Face Space.

---

## 🎯 Two flavours, one workflow

|  | 🐍 **Python app** | 🌐 **Web / JS app** |
|---|---|---|
| **Runs on** | Your laptop or the robot | Anyone's browser, from anywhere |
| **Setup** | `pip install`, daemon | Zero install — just share a URL |
| **Best for** | Tight control loops, motion sequences, on-robot logic | Remote control, dashboards, mobile UIs, quick demos |
| **Shipped as** | Python package + Hugging Face Space | Static Hugging Face Space |

Your agent picks the right flavour for your idea — or you can ask for a specific one up front.

---

## 💡 Example prompts to spark ideas

> *Build a Reachy Mini app that follows my face with the camera and wiggles an antenna when I smile.*

> *Make a **web app** where I can drive Reachy Mini from my phone — head joystick on the left, sound buttons on the right.*

> *I want Reachy Mini to play a wake-up dance every morning at 8 AM.*

> *Build a multiplayer game where Reachy Mini's head is the ball and two players control the antennas as paddles.*

---

## 📚 Going deeper (optional)

If your agent (or you) wants to dive in:

- 🤖 **[AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md)** — the agent's playbook
- 🐍 **[Python SDK reference](../source/SDK/python-sdk.md)**
- 🌐 **[JavaScript SDK reference](../source/SDK/javascript-sdk.md)**
- 📂 **[Code examples](https://github.com/pollen-robotics/reachy_mini/tree/main/examples)**
- 🎬 **[Reference web app](https://huggingface.co/spaces/cduss/webrtc_example)** — full WebRTC dashboard you can fork

---

## 💬 Share what you built

- [Discord](https://discord.gg/Y7FgMqHsub) — show off your apps, ask questions
- [Hugging Face Spaces](https://huggingface.co/spaces?q=reachy_mini) — discover community apps
- [GitHub Discussions](https://github.com/pollen-robotics/reachy_mini/discussions) — feature requests & bug reports
