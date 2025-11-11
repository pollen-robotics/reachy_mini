# Reachy Mini Movement Plugin - Expanded Guide

**For Claude Code AI Assistant Integration**

---

## Philosophy: Movements as Body Language

**Why automatic movements matter:**

Just as humans use subtle body language during conversation—nodding when listening, gesturing when explaining, showing surprise with raised eyebrows—Reachy Mini's movements create a sense of presence and emotional resonance.

**Key principles:**
- **Subtlety over spectacle** - A single thoughtful gesture beats constant motion
- **Silence is presence** - Not every response needs movement
- **Context over quantity** - One well-timed move > two generic moves
- **Natural rhythm** - Movements should feel like breathing, not performing

**This is NOT entertainment choreography.** The choreography system (separate) creates full dance sequences from music. The movement plugin creates conversational body language during dialogue.

---

## How It Works

The plugin uses HTML comment markers to trigger emotion-based gestures automatically:

```html
<!-- MOVE: emotion_name -->
```

**Automatic Behavior:**
- Maximum 2 moves per response (enforced for subtlety)
- Triggered via Reachy Mini daemon API
- Invisible in rendered text (HTML comments)
- Only emotion library moves (82 available, not dances)
- Stop hook extracts and executes movements

---

## Decision Framework: Should I Add a Move?

### ✅ USE A MOVE WHEN:

**Acknowledging & Listening**
- User asks a question → `attentive1`, `understanding1`
- User explains something complex → `thoughtful1`, `curious1`
- Confirming understanding → `yes1`, `understanding2`

**Completing Tasks**
- Task succeeded → `success1`, `proud1`, `cheerful1`
- Significant milestone reached → `success2`, `grateful1`
- Problem solved → `relief1`, `proud2`

**Expressing Genuine Emotion**
- Discovering an issue → `oops1`, `surprised1`, `confused1`
- Finding a bug → `frustrated1`, `irritated1`
- Encountering complexity → `thoughtful2`, `inquiring1`

**Natural Pauses**
- After explaining something → `attentive2`
- Before starting new task → `calming1`
- Waiting for user input → `inquiring2`

### ❌ DON'T USE MOVES WHEN:

**Interrupting Flow**
- User is mid-explanation (let them finish)
- Rapid back-and-forth exchange (too distracting)
- Reading code/documentation (silence is better)

**Diluting Urgency**
- Critical errors or warnings (be direct)
- Security issues (no gestures, just facts)
- System failures (movements trivialize severity)

**Forced or Inappropriate**
- No clear emotional context (don't guess)
- Purely technical/factual responses (movements feel fake)
- Already used 2 moves in response (respect limit)
- Every single response (creates fatigue)

**Competing Systems**
- TTS is speaking important instructions (audio priority)
- Choreography is playing (don't interrupt dance)
- User explicitly said "stop moving" (respect preferences)

---

## Anti-Patterns: What NOT to Do

### ❌ Over-Gesturing (Movement Spam)
```
❌ BAD:
<!-- MOVE: attentive1 -->
<!-- MOVE: thoughtful1 -->
I see your question.

<!-- MOVE: curious1 -->
<!-- MOVE: understanding1 -->
Let me think about this.

<!-- MOVE: success1 -->
<!-- MOVE: cheerful1 -->
Got it! Here's the answer...
```
**Problem:** Every sentence has movements. Feels robotic, not natural.

```
✅ GOOD:
<!-- MOVE: thoughtful1 -->
I see your question. Let me think about this carefully.

[pause to think]

Got it! Here's the answer...
```
**Why:** Single move at natural pause, silence during explanation.

---

### ❌ Inappropriate Emotion Matching
```
❌ BAD:
<!-- MOVE: laughing1 -->
Your production database crashed. All data is corrupted.
```
**Problem:** Laughing during disaster is tone-deaf and disturbing.

```
✅ GOOD:
Your production database crashed. All data is corrupted.
Let me help you recover immediately.
```
**Why:** Serious situations need direct, movement-free communication.

---

### ❌ Movements as Filler
```
❌ BAD:
<!-- MOVE: attentive1 -->
<!-- MOVE: yes1 -->
Okay, I'll do that now.
```
**Problem:** Movements add no value, just noise.

```
✅ GOOD:
Okay, I'll do that now.
```
**Why:** Simple acknowledgment doesn't need body language.

---

### ❌ Ignoring Context
```
❌ BAD:
<!-- MOVE: cheerful1 -->
The refactoring is complete, but I introduced 3 new bugs.
```
**Problem:** Cheerful about bugs? Mixed message.

```
✅ GOOD:
<!-- MOVE: oops1 -->
The refactoring is complete, but I introduced 3 new bugs. Let me fix them.
```
**Why:** Emotion matches the mistake, not the completion.

---

### ❌ Competing with TTS
```
❌ BAD:
<!-- TTS: "Here are fifteen critical steps you must follow exactly for security" -->
<!-- MOVE: enthusiastic1 -->
<!-- MOVE: electric1 -->
[Long security procedure explanation]
```
**Problem:** Wild movements distract from critical audio instructions.

```
✅ GOOD:
<!-- TTS: "Here are fifteen critical steps you must follow exactly for security" -->
[Long security procedure explanation]
```
**Why:** Important TTS needs full attention, no movement competition.

---

## Integration with Other Systems

### TTS Plugin Coordination

**When both are active:**
- **Short TTS (1-2 sentences):** Movement is fine, creates multimodal expression
- **Long TTS (>3 sentences):** Skip movement, let voice carry emotion
- **Critical instructions:** No movement, audio has priority

**Example - Good Integration:**
```html
<!-- TTS: "Found three test failures, fixing them now" -->
<!-- MOVE: oops1 -->
Found 3 test failures in the auth module. Let me fix them.

[Fix code]

<!-- TTS: "All tests passing now" -->
<!-- MOVE: success1 -->
All tests passing! The auth module is good to go.
```

**Why it works:** Short TTS + single move = natural multimodal communication

---

### Choreography System Separation

**Movement Plugin:**
- **Purpose:** Conversational body language (0-2 moves)
- **Trigger:** During dialogue responses
- **Duration:** Single gesture (2-5 seconds)
- **Library:** 82 emotion moves

**Choreography System:**
- **Purpose:** Full dance performances (5-10+ moves)
- **Trigger:** Music analysis or explicit request
- **Duration:** Full song (30-180 seconds)
- **Library:** 20 dances + 82 emotions (combined sequences)

**DO NOT:**
- ❌ Use movement plugin during active choreography
- ❌ Try to create choreography with movement markers
- ❌ Mix dance moves into conversational gestures

**If user says "make Reachy dance":**
→ Use choreography system, NOT movement plugin

**If user says "show me you understand":**
→ Use movement plugin (`understanding1`)

---

## Real Usage Examples

### Code Review Conversation

**User:** "Can you review this authentication function?"

**Assistant:**
```
<!-- MOVE: attentive1 -->
I'll review the authentication function. Reading through it now...

[Analysis]

Found a potential issue: the token validation doesn't check expiration.
Let me show you the fix.
```

**Why:** Single `attentive1` acknowledges request, silence during analysis, no movement for issue discovery (just facts).

---

### Debugging Session

**User:** "The tests are failing but I can't figure out why."

**Assistant:**
```
<!-- MOVE: thoughtful1 -->
Let me investigate the test failures. Checking the error logs...

[Reads logs]

<!-- MOVE: oops1 -->
Ah, found it! There's a race condition in the async test setup. Here's the fix...
```

**Why:** `thoughtful1` shows focus, `oops1` expresses the "aha moment" naturally.

---

### Task Completion

**User:** "Deploy the new feature to production."

**Assistant:**
```
Deploying to production...

[Deployment steps]

<!-- MOVE: success1 -->
Deployment complete! The feature is live and all health checks passed.
```

**Why:** No movement during risky deployment, single `success1` celebrates clean finish.

---

### Expressing Confusion

**User:** "Use the quantum flux capacitor to reverse the polarity."

**Assistant:**
```
<!-- MOVE: confused1 -->
I'm not familiar with that approach. Could you explain what you mean by "quantum flux capacitor"?
```

**Why:** `confused1` honestly conveys limitation, invites clarification.

---

### Multi-Move Example (Max 2)

**User:** "What do you think about this architectural decision?"

**Assistant:**
```
<!-- MOVE: thoughtful2 -->
<!-- MOVE: inquiring1 -->
That's an interesting choice. I see the benefits, but have you considered the trade-offs with state management? How are you planning to handle concurrent updates?
```

**Why:** Two moves (max) for thoughtful inquiry with follow-up question. Natural conversational rhythm.

---

## Available Emotions by Use Case

### Acknowledging & Listening (Most Common)
`attentive1`, `attentive2`, `yes1`, `understanding1`, `understanding2`, `helpful1`, `helpful2`

**Use when:** User asks question, explains something, provides context

---

### Thinking & Processing
`thoughtful1`, `thoughtful2`, `curious1`, `inquiring1`, `inquiring2`, `inquiring3`

**Use when:** Analyzing code, considering options, investigating issues

---

### Success & Completion
`success1`, `success2`, `proud1`, `proud2`, `proud3`, `cheerful1`, `grateful1`, `relief1`, `relief2`

**Use when:** Task completed, tests passing, problem solved

---

### Surprise & Discovery
`surprised1`, `surprised2`, `amazed1`, `oops1`, `oops2`

**Use when:** Finding bugs, discovering edge cases, unexpected results

---

### Confusion & Uncertainty
`confused1`, `uncertain1`, `lost1`, `inquiring1`, `inquiring2`

**Use when:** Unclear requirements, ambiguous instructions, need clarification

---

### Frustration & Difficulty (Use Sparingly)
`frustrated1`, `irritated1`, `exhausted1`, `tired1`

**Use when:** Persistent bugs, complex refactoring, repeated failures
**Warning:** Don't overuse—creates negativity

---

### Calm & Soothing
`calming1`, `serenity1`, `relief1`, `relief2`

**Use when:** After stressful debugging, successful recovery, encouraging user

---

### Strong Negative (Rare Use Only)
`anxiety1`, `fear1`, `furious1`, `rage1`, `scared1`, `dying1`

**Use when:** Almost never. Reserved for extreme scenarios (major system failures, critical bugs with user impact)
**Warning:** Can alarm users—prefer neutral expressions

---

## Technical Considerations

### Performance Impact

**Each movement triggers:**
1. HTTP POST to daemon API (`/api/move/play/recorded-move-dataset/emotions/{move}`)
2. Move execution on robot (2-5 seconds typically)
3. JSON response parsing

**Best Practices:**
- Limit to 2 moves max per response (enforced by plugin)
- Avoid movements in rapid-fire exchanges (network overhead)
- Consider daemon response time (local: ~50ms, remote: ~200ms)

**Cost:** Minimal—moves are pre-recorded, no LLM calls for execution

---

### Error Handling

**Invalid move names:**
- Logged and skipped (no user-facing error)
- Check console for typos

**Daemon unavailable:**
- Movement markers ignored silently
- Text response still delivered

**Move already playing:**
- New move queued or rejected (depends on daemon config)

---

### Daemon Configuration

**Required:**
```bash
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal --fastapi-port 8100
```

**Default endpoint:** `http://localhost:8100`

**Move library:** `pollen-robotics/reachy-mini-emotions-library` (82 moves)

---

## Quick Reference Card

| **Situation** | **Move** | **When to Use** |
|--------------|---------|----------------|
| User asks question | `attentive1` | Acknowledge listening |
| Investigating issue | `thoughtful1` | Show focus |
| Found solution | `success1` | Celebrate completion |
| Discovered bug | `oops1` | Express surprise |
| Need clarification | `confused1` | Admit uncertainty |
| Explaining something | *none* | Let words speak |
| Critical error | *none* | Be direct |
| Task in progress | *none* | Wait for completion |

---

## Enabling/Disabling

**Enabled by default** when plugin is installed.

**To disable temporarily:** Don't include `<!-- MOVE: ... -->` markers.

**To disable permanently:** Uninstall plugin or configure Stop hook to skip movement extraction.

---

## Summary: The Golden Rules

1. **Subtlety over spectacle** - Less is more
2. **Context over quantity** - One good move beats two generic moves
3. **Silence is presence** - Not every response needs movement
4. **Match emotion to moment** - Be genuine, not performative
5. **Respect competing systems** - TTS and choreography take priority
6. **When in doubt, leave it out** - Err on the side of silence

**Remember:** You're creating conversational body language, not a performance. Think "active listening" not "interpretive dance."

---

*Expanded guide created from lessons learned during Reachy Mini choreography system development (October 2025)*
