# Reply to feedback - conversation design

_Just sharing my opinion on what's been published in this channel. Two things seem essential enough to clarify: the **product vision** (mobile app on a single HF speech-to-speech backend, commercial adapters kept cheaply on the daemon) and the **architecture** (daemon-owned session + thin JS client over one protocol, same on Lite / Wireless / local). The point-by-point below answers Alina's note; the v1-vs-later list at the end is for Andy's scope question._

Intent: clarify, not argue.

A gentle reminder: these docs were always a **draft to discuss**, not a final spec (the overview opens with "early draft, my own opinion, not a settled decision"). So rather than starting new documents, please feel free to treat these as yours - edit, debate and rework them directly. That's exactly what they're for.

On the "hard to read" feedback: fair. The density is partly deliberate - these are reference/contract docs, meant to be precise, not skimmed - but anything genuinely unclear or inaccurate is worth flagging. And if a shorter, more concise design doc would help everyone converge faster, please go ahead and write one - I'm all for it.

Treat the substance below as the argument, not the final form.

---

**1. Conversation app as the main interface**
Fully agreed - it's the premise the whole design is built on.

**2. Keeping OpenAI and Gemini**
Split it into product vs daemon and I think we agree. **Product:** the app centers on a single HF speech-to-speech backend we own. That lets us fine-tune personalities, voice and turn-taking once and ship one consistent experience, instead of tuning three platforms - two of which we don't control and can change overnight. You can't build a product on backends that shift under you, and it grows our own backend rather than spreading the effort. **Daemon / code:** you're right the OpenAI/Gemini adapters are isolated and nearly free to keep, so we leave them in the daemon, reachable through the protocol - just not as the app's product backend. The option stays open (incl. the Gemini / Google-on-RM angle) without diluting the product.

**3. The two "inaccurate" points**
(a) WebRTC: agreed, nothing in the design assumes a custom backend - the transport is backend-agnostic. (For context, `language` in the config is just a soft hint that biases input transcription so it doesn't drift on short utterances; generic, not a lock.) (b) French-accent / user-geography: outside what I was around for, so I'll leave it to whoever has the context.

**4. Pulling conversation into the daemon**
Agreed, and the design already does this: the daemon owns the session, realtime audio stays on the robot, and the client is a thin adapter that drives/observes over the protocol. We're explicitly *not* merging the whole app into the daemon.

**5. Why a JS/TS client**
The intent is to move away from Python-apps-on-the-Pi as the authoring surface (hard to build and maintain outside Pollen). Conversation becomes a daemon service behind a stable protocol, driven by a thin JS/web client.

On "it only works on Wireless, not Lite": that's not a transport limitation. The JS SDK drives the robot over WebRTC (commands on the data channel, audio/video on media tracks) with pluggable signaling - the central HF Space by default, or a local endpoint via `?signaling_url=`. So Lite or local setups already work by pointing the SDK at local signaling; the "wireless-only" note in the docs is about the default central onboarding path (robots registered via the mobile app live on the central Space), not WebRTC itself. The daemon also unifies the hardware - Wireless and Lite "work the same way" behind it. So it's the same client and protocol everywhere; only the daemon's host and signaling endpoint change:

```
              JS / Web client  (same SDK + protocol everywhere)
                         |
            stable protocol over WebRTC DataChannel
                         |
   +---------------------+---------------------+
   |                     |                     |
 Wireless              Lite               Local / sim
 daemon on Pi      daemon on host PC   daemon on dev machine
   |                     |                     |
 robot HW          robot HW (USB)        simulated robot
```

The same SDK call (`conversation.start`, `say`, ...) behaves identically wherever the daemon runs - Pi (Wireless), host machine (Lite), or dev machine (sim); the client never needs to know. Not hypothetical: on Lite/desktop the daemon already runs on the host as a background service (`reachy_mini_tray` and the desktop app spawn the same upstream daemon and register it on the WebRTC relay), so a phone or web client reaches it exactly like a Wireless robot.

This also gives two extension paths instead of the Python-app model: (a) ship an MCP tool-space (hosted on Spaces) that shows up in `config.tools` for the model to call, or (b) build an app on the conversation primitives with the JS SDK - either way, no Python package to flash on the robot. Bigger picture: standardizing on WebRTC for any robot, local or web, is what unifies everything and makes agent-driven app creation far easier at every stage (creation, development, sharing) against one stable surface.

**6. "Create" feature / Lovable**
Agreed the building blocks already exist (tool plugins, personalities, memory), so "create" is mostly UI on top, not a rebuild - it's exactly the two extension paths above. And the Lovable-style agent-driven app creation is precisely the end goal we're aiming for here.

**7. One daemon for all robots**
That's exactly the direction: one WebRTC protocol means a single daemon surface can serve every robot, old and new, with apps attaching cleanly. Hardware differences stay behind the daemon, not in the client.

---

## Scope: v1 vs later (for Andy - where's the v1 line?)

The split already lives in the design docs' "Scope (v1 vs later)" section, but it's easy to miss in the prose. The key point: the contract is intentionally broad, but the **v1 shipping cut is small** - we are not rewriting everything for EOW.

**v1 (ship first)**
- RPCs: `start` / `stop` / `restart` / `status` / `say` / `interrupt`
- Events: `phase`, `turn`, transcripts, `level`
- `local` audio (mic + speaker stay on the robot)
- Built-in daemon tools only (fixed set: `move_head`, `play_emotion`, `dance`, `look`)
- Config: `prompt`, `voice`, `language`, `animations`, `wobble`

**Later (designed now so the protocol never has to change - but not v1)**
- `remote` / `mic_client` audio routes ("call your robot from afar")
- MCP / `tool-spaces` remote tools (a working prototype already landed)
- NFC "character card" configs and other on-robot triggers (wake-word)
- `vision.gaze` (local face-tracking -> head motion)
- Long-term memory (a separate daemon subsystem, outside this contract)

Everything in "later" is reference-based and optional, so an older robot simply ignores what it doesn't know. That's why v1 can stay small without painting us into a corner - the breadth is in the design, not in the first cut.

A caveat on timing: even this small cut may well take more than a week. I'd rather under-promise and ship something solid than rush the contract to hit EOW. Keeping the v1 cut small is precisely what makes a realistic, non-EOW timeline acceptable.

---

## On method: docs -> tests -> agent

On Rémi's point that the implementation will suck if the design docs aren't rock solid - I agree, and I'd push it one step further: a solid contract is exactly what makes this de-riskable, and it converges with the process Rémi proposed (team converges on docs, then one person + agent grinds, then team tests).

`conversation-public-api.md` is structured enough to act as a spec, not just prose: RPCs, events, `config`, error codes. That means we can turn it into a **conformance/contract test suite first**, then have the agent implement to green. The doc becomes executable; the tests pin the contract so the implementation can't drift, whoever (human or agent) writes it. That's also why a single doc-first surface matters more than raw context size - the agent codes against tests, not against the whole repo held in its head.

This works cleanly for the **control plane** (RPC / events / `config` are deterministic and mockable). The expressive, embodied parts - motion fusion, turn-taking feel, voice, S2S latency - aren't unit-testable and need human-in-the-loop iteration. That's a big part of why I think one week is probably a bit short.

---

## On Alina's design doc - two points to converge on

Her draft is a great base: concise and grounded in the existing code, exactly what we needed, and I agree with most of it (robot-hosted, keeping the backends, the presence layer, customisation already exists). Two architectural points are worth aligning on:

1. **What we expose, not where the code lives.** Keeping the engine in Python is fine by me - it's the pragmatic, EOW-friendly path. The one thing I'd hold onto: what clients touch should stay a **stable, language-agnostic protocol**, not a Python API. Engine in Python on the inside, a protocol contract on the outside - that keeps her consolidation *and* the "no Python to flash, agent-authorable" goal.

2. **Stateless contract over live reconcile.** The patch/reconcile state document is nice UX, but it's harder to specify and test, and it breaks the stateless contract that makes the "docs -> tests -> agent" path work. I'd keep the contract stateless: an atomic `restart` already swaps config without flicker, which covers most of the mid-session need. We can still allow patches for trivially live-safe fields (e.g. volume); everything else goes through restart. Same UX where it's cheap, testability preserved.

---

On the docs feeling immature: please don't hesitate to point out the specific spots that read as unclear or inaccurate to you - the more precise, the better. Wherever it falls within my scope, I'll happily steer you on it. I'll try to take ~30 min to reply while I'm away, with roughly a one-day delay each time.
