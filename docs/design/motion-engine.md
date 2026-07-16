# Motion Engine — a robot that stays alive on its own

Reachy Mini should never look like a statue. The moment it's awake it breathes,
and between two animations it drifts back to a neutral pose and keeps breathing.

The important part: **all of this runs on the robot, inside the daemon.** You don't
stream a 30 Hz "keep it alive" loop from your app, and nothing ever lags behind the
network. You send intentions ("play this emotion", "look here"), the robot handles
the life.

## The mental model: four layers

```
  offsets   (additive)     ← subtle live nudges layered on top: speech sway
  ─────────────────────
  gaze      (aim)          ← "look here": head-tracking targets, smoothed + held
  ─────────────────────
  moves     (primary)      ← emotions, gotos: you trigger them, one at a time
  ─────────────────────
  idle life (breathing)    ← the default floor when nothing else is happening
```

- **Idle life** is the floor. When no move is playing and nobody's driving, the
  daemon breathes (gentle head bob + antenna sway) around the neutral pose.
- **Moves** sit on top. Trigger one and it takes over the body; when it ends, the
  robot eases back to base and breathing resumes. No cleanup on your side.
- **Gaze** aims the head at a target you feed it. It coexists with breathing (the
  robot looks at you *and* breathes) and steps aside for moves. See below.
- **Offsets** ride on top of everything, added just before the robot moves. Great
  for "always-on" reactive motion that shouldn't fight the rest.

You'll spend 90% of your time on the **moves** layer. The rest mostly takes care
of itself.

## Looking at things (head tracking)

Head tracking is a great example of the split: **the eyes are yours, the neck is
the robot's.** Your app runs the vision (face detection, whatever) and just tells
the robot *where* to look. The daemon decides how that lands on the body.

```js
// Your vision loop, at whatever rate it runs:
onFaceDetected(({ yaw, pitch }) => robot.lookAt({ yaw, pitch }));

// Done tracking — head eases back to neutral, breathing carries on.
robot.lookAt(null);
```

What the engine does with that, so you don't have to think about it:

- **It keeps breathing underneath.** Tracking aims the head; the breathing bob
  still rides on top. The robot looks alive *while* it looks at you.
- **It's smoothed and held on the robot.** Push targets as fast or as slow as your
  detector runs — the daemon interpolates between them. If your stream stutters or
  stalls, it holds the last gaze instead of snapping to neutral. Jittery vision
  does not mean a jittery neck.
- **It yields to moves.** When an emotion plays, the emote owns the head; tracking
  pauses and picks back up when the move ends. A "look here" target never drags an
  animation off its choreography.
- **It does *not* count as "manual control".** Unlike streaming raw poses, a gaze
  stream won't switch idle life off — breathing stays alive underneath the whole
  time. Stop tracking and the head just eases home.

The only thing that's on you: because the *target* comes from your app, a bad link
makes the robot look **late**, not jittery. The smoothing hides jitter; it can't
invent data it hasn't received yet.

## Quick start

```js
const robot = new ReachyMini();
await robot.connect();
await robot.startSession(robotId);

// Wake it up. From here it breathes on its own — you do nothing.
await robot.wakeUp();

// Play an emotion. Breathing pauses, the emote plays, then it settles
// back to base and breathing picks up again.
robot.playRecordedMove("curious1");
```

That's the whole "make it feel alive" story. There is no step 4.

## The behaviour contract

A few rules worth knowing, because they're what make it feel right:

- **Alive by default.** After `wakeUp()` the robot breathes until told otherwise.
  No flag to set.
- **A move always wins over breathing.** Starting a move stops the breathing,
  plays, and hands the body back when it's done.
- **Coming back to base is free.** The breathing itself eases from wherever the
  last move left the head down to neutral, then breathes there. You never have to
  send a "go home" pose between animations.
- **Your direct commands win, and auto-resume.** If you stream your own poses
  (`setTarget`, `setHeadRpyDeg`, `setAntennas`), the engine steps aside while
  you're driving and resumes breathing a fraction of a second after you stop.
- **Latency never shows.** Breathing, the return-to-base, and the audio-reactive
  head sway all run on the robot. A laggy link makes your *commands* late, never
  the idle life.

## Public API

Everything below is a one-liner over the data channel via the SDK.

| You want to…                        | SDK call                              | What the daemon does                                              |
| ----------------------------------- | ------------------------------------- | ---------------------------------------------------------------- |
| Wake / sleep                        | `wakeUp()` / `gotoSleep()`            | Plays the trajectory, powers motors on/off, starts/stops idle    |
| Play an emotion (motion + sound)    | `playRecordedMove(name)`              | Interrupts breathing, plays, returns to base, resumes            |
| Move the head / antennas directly   | `setHeadRpyDeg(...)` / `setAntennas(...)` | Applies it, pauses idle while you drive, resumes when you stop |
| Track / look at something           | `lookAt(target)` / `lookAt(null)`     | Aims the head at a target (smoothed, held), breathing stays on; `null` releases |
| Turn idle life on/off               | `setIdle(enabled)`                    | Take manual ownership, or hand it back to the robot              |
| Freeze the ears (attentive look)    | `setListening(enabled)`               | Holds antennas still, then blends them back smoothly             |
| Layer a live offset on top          | `setOffsets(channel, ...)`            | Adds it before IK, without disturbing moves or breathing         |

> `setIdle`, `setListening` and `setOffsets` are optional. Reach for them only when
> the defaults aren't enough — most apps never do.

## When to take the wheel

The one thing to keep in mind: **if you run your own continuous motion loop from
the app** (your own breathing, your own sway), call `setIdle(false)` first so the
two don't push against each other. Otherwise, leave idle on and let the robot be
alive for you — that's the whole point.

## Why it lives in the daemon

Because "alive" is a property of the robot, not of your app. Every app — the mobile
onboarding, a conversation, a game, someone's weekend hack — gets the same calm,
breathing, never-frozen Reachy for free, and none of them has to re-implement a
motion loop or fight the network to do it.