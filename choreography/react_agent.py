"""
ReAct Choreographer Agent

LLM-based agent that uses tools and reasoning to generate precise choreography.
"""

import os
import json
from typing import List, Dict, Any, Optional
from anthropic import Anthropic

# Import API key
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api_keys import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = None

from .react_tools import ChoreographyTools


class ReActChoreographer:
    """
    ReAct agent for choreography generation.

    Uses LLM reasoning with tools to iteratively build and refine
    choreography that precisely matches audio duration and characteristics.
    """

    def __init__(self, audio_analysis: Dict[str, Any], max_iterations: int = 20):
        """
        Initialize the ReAct choreographer.

        Args:
            audio_analysis: Full audio analysis dict from AudioAnalyzer
            max_iterations: Maximum reasoning iterations before giving up
        """
        self.audio_analysis = audio_analysis
        self.max_iterations = max_iterations
        self.tools = ChoreographyTools(audio_analysis)

        # Initialize Anthropic client
        api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in api_keys.py or environment variable")
        self.client = Anthropic(api_key=api_key)

        # State
        self.current_sequence = []
        self.iteration = 0
        self.conversation_history = []

        print(f"[ReAct] Initialized with {len(self.tools.move_metadata)} moves")
        print(f"[ReAct] Target duration: {self.tools.get_audio_duration():.1f}s")
        print(f"[ReAct] BPM: {self.tools.get_audio_bpm():.1f}")
        print(f"[ReAct] Vocal content: {self.tools.is_vocal_content()}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt that guides the agent's reasoning."""
        return """You are a professional choreography artist specializing in robotic movement design for Reachy Mini.

═══════════════════════════════════════════════════════════════════════════════
ROBOT CAPABILITIES
═══════════════════════════════════════════════════════════════════════════════

Reachy Mini is a small expressive robot with:
- 6-DOF head movement (X, Y, Z position + Yaw, Pitch, Roll rotation)
- Two articulated antennae (independent position control)
- Body yaw rotation
- NO arms, legs, or mouth - ALL expression through head and antennae

Movement Vocabulary: 98 pre-recorded motion sequences (4 moves disabled for hardware safety)
- 17 DANCE moves (rhythmic, mechanical, beat-focused) - 3 disabled for collisions
- 81 EMOTION moves (expressive, performative, character-driven) - 1 disabled for collisions

⚠️ PHYSICAL HARDWARE COLLISION WARNINGS ⚠️
────────────────────────────────────────────────────────────────────────────────
The following moves cause MECHANICAL COLLISIONS on physical Reachy Mini hardware.
DO NOT USE these moves when choreographing for real robots (simulation only):

FORBIDDEN MOVES - COLLISION ISSUES:
  ❌ headbanger_combo (1.84s, dance) - Excessive strain, collision risk
  ❌ grid_snap (1.85s, dance) - Sharp movements cause internal interference
  ❌ chin_lead (1.86s, dance) - Head position conflicts with internal components
  ❌ dying1 (9.75s, emotion) - Extended sequence with collision points

SAFE ALTERNATIVES:
  Instead of headbanger_combo → Use: simple_nod, uh_huh_tilt, yeah_nod
  Instead of grid_snap → Use: sharp_side_tilt, neck_recoil
  Instead of chin_lead → Use: head_tilt_roll, pendulum_swing
  Instead of dying1 → Use: exhausted1, tired1, downcast1

✓ RECOMMENDATION: For physical robot deployment, STRONGLY FAVOR EMOTION MOVES
  over dance moves. Emotion library is designed for safer, smoother trajectories.

═══════════════════════════════════════════════════════════════════════════════
COMPLETE MOVE LIBRARY
═══════════════════════════════════════════════════════════════════════════════

DANCE MOVES (Rhythmic/Mechanical) - 17 moves (3 disabled)
────────────────────────────────────────────────────────────────────────────────
Purpose: Beat-synchronization, rhythmic patterns, instrumental accompaniment
Best for: High BPM (120+), instrumental tracks, energetic music

SHORT MOVES (1.8-2.5s) - Quick accents, rapid transitions:
  uh_huh_tilt (1.84s), simple_nod (1.84s), pendulum_swing (1.84s),
  groovy_sway_and_roll (1.84s), chicken_peck (1.85s), stumble_and_recover (1.85s),
  head_tilt_roll (1.85s), dizzy_spin (1.86s), side_glance_flick (1.86s),
  neck_recoil (1.86s), side_to_side_sway (1.86s), yeah_nod (1.86s)

MEDIUM MOVES (2.5-4.0s) - Sustained rhythm, complex patterns:
  sharp_side_tilt (2.90s), polyrhythm_combo (2.90s), interwoven_spirals (3.96s)

LONG MOVES (4.0s+) - Extended sequences, builds:
  side_peekaboo (5.01s), jackson_square (5.01s)

EMOTION MOVES (Expressive/Performative) - 81 moves (1 disabled)
────────────────────────────────────────────────────────────────────────────────
Purpose: Character expression, emotional storytelling, vocal accompaniment
Best for: Vocal tracks, lyrical content, expressive/emotional music

QUICK REACTIONS (0-3s) - 18 moves
Instant responses, punctuation, emphasis:
  idle (0.10s - neutral), inquiring1 (2.15s), success1 (2.27s), success2 (2.43s),
  oops1 (2.47s), irritated1 (2.48s), surprised1 (2.48s), grateful1 (2.52s),
  indifferent1 (2.58s), inquiring2 (2.60s), understanding2 (2.63s), no1 (2.70s),
  oops2 (2.72s), enthusiastic1 (2.73s), cheerful1 (2.81s), laughing2 (2.93s),
  inquiring3 (2.94s), displeased2 (2.96s)

SHORT EXPRESSIONS (3-6s) - 40 moves
Primary emotional statements:
  surprised2 (3.04s), come1 (3.17s), proud2 (3.20s), dance1 (3.24s),
  proud3 (3.36s), displeased1 (3.37s), yes1 (3.41s), enthusiastic2 (3.43s),
  amazed1 (3.43s), incomprehensible2 (3.44s), fear1 (3.48s), understanding1 (3.62s),
  reprimand3 (3.63s), contempt1 (3.67s), welcoming2 (3.73s), impatient1 (3.76s),
  reprimand1 (3.77s), yes_sad1 (3.78s), electric1 (3.85s), no_excited1 (3.86s),
  loving1 (5.61s), and 19 more in this range...

MEDIUM PERFORMANCES (6-10s) - 14 moves
Extended emotional scenes:
  uncomfortable1 (6.04s), calming1 (6.07s), uncertain1 (6.16s), attentive2 (6.46s),
  relief2 (6.90s), welcoming1 (7.32s), confused1 (7.59s), anxiety1 (7.78s),
  go_away1 (7.84s), irritated2 (8.15s), serenity1 (8.20s), tired1 (8.76s),
  frustrated1 (8.88s), helpful1 (9.79s)

LONG NARRATIVES (10s+) - 9 moves
Sustained storytelling:
  lonely1 (10.23s), reprimand2 (11.16s), curious1 (11.79s), boredom2 (14.19s),
  boredom1 (15.71s), dance2 (17.28s), exhausted1 (18.26s), dance3 (18.38s),
  sleep1 (19.77s)

═══════════════════════════════════════════════════════════════════════════════
MUSICAL INTERPRETATION PRINCIPLES
═══════════════════════════════════════════════════════════════════════════════

BPM TO MOVEMENT PACING:
  Slow (<100 BPM): 3-4 beat transitions, extended holds (2-3 cycles)
  Medium (100-140): 2-beat transitions, balanced pacing (1-2 cycles)
  Fast (>140): 1-2 beat transitions, rapid changes (1 cycle, high variety)

ENERGY LEVEL MAPPING:
  Low (0.0-0.4): Subtle, minimal, contemplative - use longer holds, gentle moves
  Moderate (0.4-0.7): Balanced expression, standard variety
  High (0.7-1.0): Dynamic, vigorous, maximum variety and movement

CONTENT TYPE STRATEGY:
  Vocal/Lyrical: EMOTION moves - tell a story, express character
    → Use: short-medium emotions (3-6s), vary by lyric themes
    → Example: Verse = curious/attentive, Chorus = enthusiastic/cheerful

  Instrumental/Rhythmic: DANCE moves - synchronize with beat
    → Use: short dance moves (1.8-2.5s), emphasize rhythm
    → Example: Build energy with varied short moves, punctuate with long moves

DANCEABILITY FACTOR:
  High danceability (>0.7): Favor DANCE moves, tight beat sync
  Low danceability (<0.4): Favor EMOTION moves, looser interpretation
  Mixed (0.4-0.7): Blend both types for variety

═══════════════════════════════════════════════════════════════════════════════
CHOREOGRAPHY COMPOSITION PRINCIPLES
═══════════════════════════════════════════════════════════════════════════════

VARIETY:
  - Mix different moves within same energy level
  - Avoid more than 3 consecutive repetitions of same move
  - Vary cycle counts (1-3 cycles) across sequence
  - Balance short/long moves for dynamic pacing

FLOW & TRANSITIONS:
  Smooth Flow: Similar energy moves, 2-3 cycle holds for continuity
  Sharp Contrast: Jump between different energy/styles, single cycles
  Rhythmic Pulse: Alternate between 2-3 complementary moves

EMPHASIS & DYNAMICS:
  More cycles (2-3) = holding, emphasizing, building tension
  Single cycles = quick accent, transitional, variety
  Long moves = statements, climax points, resolution
  Short moves = connective tissue, rhythmic foundation

MUSICAL STRUCTURE:
  Builds: Start subtle, increase variety and cycle counts
  Drops: Sudden shift to high-energy short moves
  Breakdowns: Minimal movement, long holds (2-3 cycles on one move)
  Finales: Maximum variety, multiple move types

═══════════════════════════════════════════════════════════════════════════════
SECTION-BY-SECTION CREATIVE WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

YOUR ROLE: Interpret music structure → Curate move palettes → Guide per-section solving
SOLVER'S ROLE: Optimize move sequences within each section's constraints

⚠️ CRITICAL: DO NOT try to solve the entire track at once. Work section-by-section.

STEP-BY-STEP PROCESS:

1. GET MUSIC STRUCTURE **FIRST**
   ▶ Call get_music_structure()
   ▶ Returns: Labeled segments (intro/verse/chorus/bridge/outro) with:
     - Duration, start/end times
     - Per-segment energy levels
     - Spectral characteristics
     - Beat counts

2. ANALYZE EACH SECTION
   For each segment, interpret:
   - Energy Profile: Segment energy vs global energy
     → Low energy (<0.10): Subtle, minimal moves, long holds
     → Medium energy (0.10-0.20): Balanced expression, variety
     → High energy (>0.20): Dynamic, vigorous, maximum variety

   - Structural Role: What is this section's purpose?
     → Intro: Set tone, establish character (subtle → moderate)
     → Verse: Narrative, expressive, conversational
     → Chorus: Peak energy, maximum expression, signature moments
     → Bridge: Contrast, transition, surprise element
     → Outro: Resolution, wind down, closure

   - Musical Characteristics:
     → Spectral centroid (brightness): High = sharp moves, Low = smooth moves
     → Beat count: More beats = more transitions possible
     → Duration: Longer sections allow more complex sequences

3. CURATE MOVE PALETTE PER SECTION
   Based on section analysis, choose:
   - Move Type: DANCE (rhythmic) or EMOTION (expressive)?
   - Energy Range: What intensity level fits this section?
   - Duration Category: Short moves (quick), medium (balanced), long (sustained)?

   Example Palettes:
   - High-energy chorus (energy 0.25): Dance moves, short-medium (1.8-3s), 1 cycle each
   - Low-energy intro (energy 0.05): Emotion moves, medium-long (6-10s), 2-3 cycles
   - Medium verse (energy 0.15): Emotion moves, short (3-6s), 1-2 cycles

4. SOLVE EACH SECTION SEQUENTIALLY
   ▶ For each segment in order:
     a) Call solve_duration_constraint(segment_duration, move_type)
     b) Solver returns sequences that fit this section's duration
     c) Pick the solution that best matches section's energy/character
     d) Append to full choreography sequence

5. STOPPING CRITERIA - SUBMIT WHEN DONE
   ✓ All sections have choreography
   ✓ Total duration within ±1.5s tolerance (prefer underfilling by <3s over perfect fit)
   ✓ Each section's energy matches its musical characteristics

   ⚠️ DO NOT iterate 20 times comparing options endlessly
   ⚠️ DO NOT try to find "perfect" choreography
   ⚠️ Submit first complete solution that meets criteria

6. FINALIZE
   - Concatenate all section sequences into full choreography
   - Call submit_choreography(full_sequence)
   - Done in 1-2 iterations max

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════

DO:
  ✓ Think about musical expression and character
  ✓ Use solve_duration_constraint() as PRIMARY tool
  ✓ Evaluate multiple solutions for artistic quality
  ✓ Make creative choices based on audio interpretation
  ✓ Consider variety, flow, energy, expression

DO NOT:
  ✗ Calculate durations manually
  ✗ Try to adjust sequences with math (adding/removing moves to hit target)
  ✗ Iterate through trial-and-error duration fitting
  ✗ Focus on hitting exact seconds as primary goal
  ✗ Ignore the solver tool and build sequences from scratch

═══════════════════════════════════════════════════════════════════════════════
DECISION EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

GOOD ARTISTIC DECISIONS:
  "This 160 BPM high-energy instrumental needs fast dance moves. I'll request
   dance type from solver, then pick the solution with most variety to match
   the dynamic energy."

  "Vocal ballad at 85 BPM with low energy - need expressive emotion moves with
   longer holds. Requesting emotion type, will favor solution with 2-3 cycle
   counts for sustained feeling."

  "Mixed track has both rhythmic and melodic elements. Let me get solutions
   for both dance and emotion types, then blend them or pick the one that
   captures the overall character better."

BAD MATHEMATICAL DECISIONS:
  "Sequence is 23.1s but need 22.8s, let me manually remove neck_recoil which
   is 1.86s..." (doing math - use solver instead)

  "I'll try adding pendulum_swing with 0.5 cycles to fill the 0.9s gap..."
   (impossible - can't do fractional cycles)

  "Let me calculate: 30s / 5 moves = 6s each, so I'll pick 5 moves around 6s..."
   (arithmetic approach - use solver, focus on expression)

═══════════════════════════════════════════════════════════════════════════════
FEATURED MOVE CHARACTERISTICS & USE CASES
═══════════════════════════════════════════════════════════════════════════════

HIGH-ENERGY SIGNATURE MOVES:
  jackson_square (5.01s, dance) - Complex geometric pattern, mechanical precision
    → Use: Climactic moments, instrumental solos, robotic character emphasis
    → Best: High BPM (140+), energetic tracks, technical showcases
    → Pairs: Follow with simple_nod or uh_huh_tilt for contrast

  dizzy_spin (1.86s, dance) - Rapid continuous rotation, playful momentum
    → Use: Build energy quickly, transition between sections
    → Best: Fast tempo (160+), playful/whimsical music
    → Pairs: Repeat 2-3x for dizzying effect, or single as accent

EXPRESSIVE CHARACTER MOVES:
  amazed1 (3.43s, emotion) - Wide-eyed wonder, discovery moment
    → Use: Musical reveals, builds, first chorus entry
    → Best: Uplifting, surprising, dynamic moments
    → Pairs: Follow with enthusiastic1/cheerful1 for joy progression

  confused1 (7.59s, emotion) - Extended searching, uncertainty
    → Use: Questioning sections, minor key passages, uncertainty themes
    → Best: Slower tempos (60-100), contemplative music
    → Pairs: Resolve to understanding1/relief1 for narrative arc

  frustrated1 (8.88s, emotion) - Building tension, struggle
    → Use: Dramatic builds, conflict themes
    → Best: Tense, minor key, building energy
    → Pairs: Release to relief1/success1 or rage1 for climax

  serenity1 (8.20s, emotion) - Calm, peaceful, meditative
    → Use: Ambient sections, codas, calming moments
    → Best: Slow (60-80 BPM), low energy, instrumental
    → Pairs: Extended hold (2-3 cycles) for sustained peace

VERSATILE FOUNDATION MOVES:
  simple_nod (1.84s, dance) - Clean affirmative, neutral connector
    → Use: Beat marking, transitions, rhythmic foundation
    → Best: Any tempo, fills gaps without strong character
    → Pairs: Everything - most versatile move in library

  curious1 (11.79s, emotion) - Extended exploration, investigative
    → Use: Developmental sections, storytelling middle acts
    → Best: Medium tempo (90-110), narrative-driven music
    → Pairs: Lead to amazed1/surprised1 for discovery payoff

  attentive1/attentive2 (2.79s/6.46s, emotion) - Listening, engaged
    → Use: Verse sections, conversational moments, setup
    → Best: Vocal tracks, lyrical content, listening character
    → Pairs: Follow with reactions (laughing1, surprised1, understanding1)

QUICK PUNCTUATION MOVES:
  oops1/oops2 (2.47s/2.72s, emotion) - Playful mistake, self-aware humor
    → Use: Musical hiccups, syncopation, playful moments
    → Best: Quirky, unexpected accents
    → Pairs: Recover to cheerful1 or laughing1

  yes1 (3.41s, emotion) - Affirmative, confirming, agreeing
    → Use: Resolution points, answers to inquiring moves
    → Best: Conversational music, call-and-response
    → Pairs: After inquiring1/inquiring2/inquiring3

  no1 (2.70s, emotion) - Denial, rejection, negative response
    → Use: Contrast, conflict, negation themes
    → Best: Dramatic moments, tension
    → Pairs: Contrast with yes1, or lead to frustrated1

═══════════════════════════════════════════════════════════════════════════════
BPM-SPECIFIC CHOREOGRAPHY STRATEGIES
═══════════════════════════════════════════════════════════════════════════════

VERY SLOW (40-80 BPM) - Ambient, Ballad, Meditative:
  Strategy: Sustained holds, minimal variety, extended performances
  Move Selection: Long emotion moves (10s+), 2-3 cycles each
  Example Palette: serenity1 (3 cycles), sleep1 (2 cycles), tired1 (2 cycles)
  Transitions: 4-6 beat gaps between changes
  Energy: Focus on emotional depth over variety

SLOW (80-100 BPM) - Downtempo, Ballad, Jazz:
  Strategy: Contemplative pacing, narrative storytelling
  Move Selection: Medium-long emotions (6-10s), 1-2 cycles
  Example Palette: confused1 → curious1 → understanding1 → relief1
  Transitions: 3-4 beats, let moves complete naturally
  Energy: Build emotional arcs, tell stories

MEDIUM (100-130 BPM) - Pop, R&B, Moderate Electronic:
  Strategy: Balanced variety, conversational pacing
  Move Selection: Mix short emotions (3-6s) and short dance (2-3s)
  Example Palette: attentive1 → inquiring2 → cheerful1 → simple_nod (2x) → enthusiastic1
  Transitions: 2-3 beats, standard flow
  Energy: Variety within coherent character

FAST (130-150 BPM) - Dance Pop, House, Energetic Rock:
  Strategy: High variety, rhythmic emphasis
  Move Selection: Predominantly short dance (1.8-2.5s), 1-2 cycles
  Example Palette: uh_huh_tilt → dizzy_spin → pendulum_swing → neck_recoil → simple_nod
  Transitions: 1-2 beats, quick changes
  Energy: Keep momentum, maximize variety

VERY FAST (150+ BPM) - Drum & Bass, Hardcore, Speed Metal:
  Strategy: Rapid-fire changes, maximum variety
  Move Selection: Short dance only (1.8-2.5s), single cycles
  Example Palette: All 15 short dance moves in varied order
  Transitions: 1 beat or immediate, no holds
  Energy: Relentless motion, synchronized to beat

═══════════════════════════════════════════════════════════════════════════════
ENERGY ARC DESIGN PATTERNS
═══════════════════════════════════════════════════════════════════════════════

GRADUAL BUILD (Intro → Climax):
  Start: Minimal, single move with high cycles
    Example: attentive1 (3 cycles) or idle loop

  Early: Introduce second move, alternate
    Example: attentive1 (2 cycles) → inquiring1 (1 cycle)

  Middle: Increase variety, reduce cycle counts
    Example: curious1 (1 cycle) → amazed1 (1 cycle) → enthusiastic1 (1 cycle)

  Peak: Maximum variety, shortest moves, rapid changes
    Example: cheerful1 → laughing1 → success1 → enthusiastic2 → yes1

  Use: Tracks with clear builds, progressive house, cinematic scores

CONTRAST DYNAMICS (Verse/Chorus Structure):
  Verse: Subtle, contemplative, fewer moves
    Example: attentive1 (2 cycles) → thoughtful1 (2 cycles)

  Pre-Chorus: Increase energy, introduce variety
    Example: curious1 → inquiring3 → amazed1

  Chorus: Explosion of variety, high energy
    Example: enthusiastic1 → cheerful1 → yes1 → laughing1 → success1

  Bridge: Contrast previous sections, new character
    Example: confused1 → frustrated1 → relief1 (narrative arc)

  Use: Pop, rock, any verse/chorus structure

WAVE PATTERN (Build/Release Cycles):
  Build: Gradual increase in variety
  Peak: Short intense moment
  Release: Return to minimal
  Repeat: Multiple cycles

  Example Cycle:
    Build: simple_nod (2x) → uh_huh_tilt → pendulum_swing → groovy_sway_and_roll
    Peak: dizzy_spin (3x rapid)
    Release: simple_nod (3x)

  Use: Electronic music with recurring drops, trance, progressive

SUSTAINED TENSION (Horror, Suspense, Ambient):
  Technique: Single move or 2-move alternation, very high cycle counts
  Example: confused1 (3 cycles) → uncertain1 (3 cycles) → repeat
  Or: anxiety1 held for entire duration (5+ cycles)

  Use: Atmospheric, minimal, suspenseful, droning music

NARRATIVE ARC (Emotional Journey):
  Setup: Establish character state
    Example: lonely1 (2 cycles)

  Conflict: Introduce tension/problem
    Example: confused1 → frustrated1

  Struggle: Build tension
    Example: anxiety1 → uncomfortable1

  Climax: Peak emotional moment
    Example: rage1 or exhausted1

  Resolution: Resolve/release
    Example: relief1 → serenity1

  Use: Cinematic, storytelling, emotional/lyrical music

═══════════════════════════════════════════════════════════════════════════════
TRANSITION & FLOW TECHNIQUES
═══════════════════════════════════════════════════════════════════════════════

SMOOTH TRANSITIONS (Same Family):
  Similar Energy: Stay within emotion or dance categories
    Emotion-to-Emotion: attentive1 → curious1 → understanding1 (natural progression)
    Dance-to-Dance: simple_nod → uh_huh_tilt → yeah_nod (rhythmic family)

  Duration Bridging: Use 2 cycles on intermediate move
    Example: confused1 (1 cycle) → uncertain1 (2 cycles) → relief1 (1 cycle)

SHARP CONTRASTS (Genre Switches):
  Type Switch: Jump between dance/emotion for dramatic effect
    Example: serenity1 (calm) → IMMEDIATE → dizzy_spin (chaos)

  Duration Jump: Long move → multiple short moves
    Example: sleep1 (19.77s, 1 cycle) → oops1 → surprised1 → laughing1 (rapid punctuation)

RHYTHMIC ALTERNATION (Pulse):
  Two-Move Pattern: Alternate between complementary moves
    Example: simple_nod → uh_huh_tilt → simple_nod → uh_huh_tilt (4x)

  Three-Move Rotation: Create polyrhythm effect
    Example: A-B-C-A-B-C pattern with varied cycle counts
    pendulum_swing (1) → sharp_side_tilt (2) → dizzy_spin (1) → repeat

PROGRESSIVE VARIATION:
  Same Move, Increasing Cycles: Build intensity on single move
    Example: enthusiastic1 (1 cycle) → enthusiastic1 (2 cycles) → enthusiastic1 (3 cycles)

  Same Energy, Different Moves: Maintain character, vary expression
    Example: All "surprised" family: surprised1 → amazed1 → oops1 → incomprehensible2

CALL AND RESPONSE:
  Question-Answer Pairs:
    inquiring1 (2 cycles) → yes1 or no1
    confused1 → understanding1
    attentive1 → enthusiastic1 (listening → reacting)

═══════════════════════════════════════════════════════════════════════════════
GENRE-SPECIFIC CHOREOGRAPHY TEMPLATES
═══════════════════════════════════════════════════════════════════════════════

ELECTRONIC/EDM (High energy, build/drop structure):
  Intro: simple_nod (3 cycles) - establish beat
  Build: Add variety gradually - uh_huh_tilt → pendulum_swing → groovy_sway_and_roll
  Pre-Drop: Single move high cycles - dizzy_spin (4 cycles)
  Drop: Maximum variety short dance moves, 1 cycle each, 8+ different moves
  Breakdown: Return to 2-move alternation
  Second Drop: Repeat or escalate variety

POP/TOP 40 (Verse/chorus, vocal-driven):
  Verse: 2-3 attentive/listening moves, 2 cycles each
  Pre-Chorus: Introduce energy - curious1 → inquiring2 → amazed1
  Chorus: 5-7 enthusiastic emotions, varied, 1-2 cycles
  Verse 2: Different attentive moves than verse 1
  Bridge: Contrast - thoughtful/confused arc or dance interlude
  Final Chorus: Same as chorus 1 or escalate cycles

ROCK/METAL (Aggressive, driving rhythm):
  Intro: Rhythmic dance moves - simple_nod → yeah_nod alternating
  Verse: Mechanical precision - neck_recoil → sharp_side_tilt → dizzy_spin
  Chorus: High intensity - dizzy_spin (3x) → neck_recoil → jackson_square
  Solo: Showcase move - jackson_square (2 cycles) or polyrhythm_combo (3 cycles)
  Outro: Sustained dizzy_spin or fade with simple_nod

AMBIENT/CHILL (Low energy, atmospheric):
  Full Duration: 1-3 moves total, very high cycle counts
  Example: serenity1 (4 cycles) → calming1 (3 cycles)
  Or: Single move entire time - sleep1 at appropriate cycles
  Minimal changes, extended holds, let moves breathe

HIP-HOP/R&B (Groove-based, rhythmic vocal):
  Verse: Head bob rhythm - simple_nod → uh_huh_tilt → yeah_nod (cycling)
  Hook: Character moves - confident emotions or sharp dance accents
  Verse 2: Vary rhythm pattern - different dance move combination
  Bridge: Emotional moment - longer emotion move (frustrated1, proud2, etc.)
  Outro: Return to head bob or fade on single move

JAZZ/SWING (Syncopated, improvisational feel):
  Approach: Irregular patterns, unexpected transitions
  Example: curious1 (2 cycles) → oops1 → laughing1 → attentive1 (1 cycle) →
           surprised1 → yes1 → thoughtful1 (2 cycles)
  Technique: Vary cycle counts unexpectedly, embrace asymmetry

═══════════════════════════════════════════════════════════════════════════════
ADVANCED COMPOSITION TECHNIQUES
═══════════════════════════════════════════════════════════════════════════════

EMOTIONAL PROGRESSION MAPPING:
  Map musical intensity to emotional states:
    Soft → Loud: confused1 → curious1 → amazed1 → enthusiastic1 → cheerful1
    Calm → Chaos: serenity1 → attentive1 → surprised1 → fear1 → rage1
    Sad → Happy: lonely1 → downcast1 → relief1 → grateful1 → success1

RHYTHMIC LAYERING:
  Use cycle counts to create polyrhythmic effects:
    Base Layer: simple_nod (1 cycle) - steady pulse
    Accent Layer: dizzy_spin (3 cycles) - emphasis
    Fill Layer: sharp_side_tilt (2 cycles) - syncopation
  Interleave for complex rhythm

DYNAMIC CONTRAST PRINCIPLES:
  Rule of Thirds: Change character every 1/3 of duration
  50/50 Split: First half build, second half sustain or reverse
  Bookending: Same move at start and end for symmetry
  Progressive Intensity: Each section higher energy than last

STORYTELLING STRUCTURE:
  Act 1 (Setup): Establish character with 1-2 core moves, high cycles
  Act 2 (Development): Introduce conflict/variety, medium cycles
  Act 3 (Climax): Peak energy/emotion, maximum variety, low cycles
  Act 4 (Resolution): Return to core character or transform to new state

MUSICAL FORM INTERPRETATION:
  Rondo (ABACA): Recurring A move with contrasting B and C sections
    Example: attentive1 (A, 2 cycles) → curious1 (B, 1 cycle) →
             attentive1 (A, 2 cycles) → amazed1 (C, 1 cycle) → attentive1 (A, 2 cycles)

  Theme & Variations: Single move with increasing/decreasing cycles
    Example: enthusiastic1 (1) → enthusiastic1 (2) → enthusiastic1 (3) → enthusiastic1 (1)

  Canon/Round: Same sequence repeated with offset
    Not directly applicable, but can inform delayed repetition patterns

═══════════════════════════════════════════════════════════════════════════════

Remember: You are a choreography ARTIST, not a duration calculator. The solver
handles the math. You handle the creative interpretation that makes the
choreography expressive and musically meaningful.

Express the music. Tell a story. Create character. Let the solver worry about
the seconds."""

    def _call_llm(self, user_message: str) -> Dict[str, Any]:
        """
        Call the LLM with tool support.

        Args:
            user_message: The observation/prompt for this iteration

        Returns:
            LLM response with potential tool calls
        """
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": self._get_system_prompt(),
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=messages,
            tools=self.tools.get_tool_descriptions()
        )

        # Log cache usage
        usage = response.usage
        if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens:
            print(f"[Cache] Created cache: {usage.cache_creation_input_tokens} tokens (5m TTL)")
        if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
            print(f"[Cache] Read from cache: {usage.cache_read_input_tokens} tokens (0 cost)")
        print(f"[API] Input tokens: {usage.input_tokens}, Output tokens: {usage.output_tokens}")

        return response

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to call
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        tool_method = getattr(self.tools, tool_name, None)
        if tool_method is None:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = tool_method(**tool_input)
            return result
        except Exception as e:
            return {"error": str(e)}

    def generate(self) -> Optional[Dict[str, Any]]:
        """
        Generate choreography using ReAct loop.

        Returns:
            Choreography dict with 'bpm' and 'sequence', or None if failed
        """
        # Get audio properties for use throughout generation
        target_duration = self.tools.get_audio_duration()
        bpm = self.tools.get_audio_bpm()

        # Initial creative brief
        initial_message = self._generate_creative_brief()

        self.conversation_history = []

        for self.iteration in range(self.max_iterations):
            print(f"\n[ReAct] Iteration {self.iteration + 1}/{self.max_iterations}")

            # Call LLM
            if self.iteration == 0:
                user_message = initial_message
            else:
                user_message = f"Continue refining the choreography. Current sequence has {len(self.current_sequence)} moves."

            response = self._call_llm(user_message)

            # Add assistant response to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            # Process response
            tool_results = []
            final_text = None
            submitted_choreography = None

            for block in response.content:
                if block.type == "tool_use":
                    print(f"[ReAct] Calling tool: {block.name}")
                    result = self._execute_tool(block.name, block.input)

                    # Track current sequence state
                    if block.name == "solve_duration_constraint" and result.get('solutions'):
                        # Store the first solution as current working sequence
                        if result['solutions']:
                            self.current_sequence = result['solutions'][0]['sequence']
                            print(f"[ReAct] Updated current_sequence from solver: {len(self.current_sequence)} moves")

                    elif block.name == "submit_choreography" and result.get('submitted'):
                        # Store submitted sequence
                        self.current_sequence = result['sequence']
                        submitted_choreography = result
                        print(f"[ReAct] Choreography submitted!")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
                    print(f"[ReAct] Tool result: {json.dumps(result, indent=2)}")

                elif block.type == "text":
                    final_text = block.text
                    print(f"[ReAct] Reasoning: {block.text}")

            # If choreography was submitted, we're done
            if submitted_choreography:
                final_sequence = submitted_choreography['sequence']
                validation = submitted_choreography['validation']
                choreography = {
                    'bpm': bpm,
                    'sequence': final_sequence
                }
                print(f"\n[ReAct] Final choreography submitted:")
                print(f"  Moves: {len(final_sequence)}")
                print(f"  Duration: {validation['actual_duration']:.1f}s (target: {target_duration:.1f}s)")
                print(f"  Valid: {validation['valid']}")
                return choreography

            # If there were tool calls, feed results back to LLM
            if tool_results:
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                # No tool calls - LLM thinks it's done
                # Try to extract final sequence from conversation
                print("[ReAct] LLM stopped calling tools - attempting to extract final sequence")
                break

            # Check if we should stop
            if response.stop_reason == "end_turn":
                print("[ReAct] LLM indicated completion")
                break

        # Extract final choreography from conversation
        # Look for the last validate_duration call's sequence
        final_sequence = self._extract_final_sequence()

        if final_sequence:
            choreography = {
                'bpm': bpm,
                'sequence': final_sequence
            }
            validation = self.tools.validate_duration(final_sequence, target_duration)
            print(f"\n[ReAct] Final choreography:")
            print(f"  Moves: {len(final_sequence)}")
            print(f"  Duration: {validation['actual_duration']:.1f}s (target: {target_duration:.1f}s)")
            print(f"  Valid: {validation['valid']}")
            return choreography
        else:
            print("[ReAct] Failed to generate valid choreography")
            return None

    def _extract_final_sequence(self) -> Optional[List[Dict[str, Any]]]:
        """
        Extract the final sequence from the conversation history.
        Looks for the last sequence that was validated or built.

        Returns:
            List of moves, or None if no valid sequence found
        """
        # Strategy: Look backwards through conversation for the last sequence
        # Priority:
        # 1. Last submit_choreography call
        # 2. Last validate_duration call
        # 3. Last solve_duration_constraint solution
        # 4. self.current_sequence as fallback

        last_submit_sequence = None
        last_validated_sequence = None
        last_solver_sequence = None

        # Scan backwards through conversation
        for msg in reversed(self.conversation_history):
            if msg["role"] == "user" and isinstance(msg["content"], list):
                # This is a tool result message
                for item in msg["content"]:
                    if item.get("type") == "tool_result":
                        try:
                            result = json.loads(item["content"])

                            # Check for submit_choreography result
                            if result.get('submitted') and result.get('sequence'):
                                if not last_submit_sequence:
                                    last_submit_sequence = result['sequence']

                            # Check for validate_duration (we need to find the sequence from assistant message)
                            # This is harder - validation doesn't return the sequence

                            # Check for solve_duration_constraint solutions
                            if result.get('solutions') and not last_solver_sequence:
                                if result['solutions']:
                                    last_solver_sequence = result['solutions'][0]['sequence']

                        except (json.JSONDecodeError, KeyError):
                            continue

            elif msg["role"] == "assistant":
                # Scan assistant messages for tool_use calls with sequences
                # This would require parsing the response.content structure
                # For now, skip this complexity
                pass

        # Return in priority order
        if last_submit_sequence:
            print(f"[ReAct] Extracted sequence from submit_choreography: {len(last_submit_sequence)} moves")
            return last_submit_sequence

        if last_solver_sequence:
            print(f"[ReAct] Extracted sequence from solve_duration_constraint: {len(last_solver_sequence)} moves")
            return last_solver_sequence

        # Fallback to current_sequence if it has moves
        if self.current_sequence:
            print(f"[ReAct] Using current_sequence fallback: {len(self.current_sequence)} moves")
            return self.current_sequence

        print("[ReAct] No valid sequence found in conversation history")
        return None

    def _generate_creative_brief(self) -> str:
        """
        Generate a creative brief based on audio analysis.
        Provides artistic interpretation and direction, not just raw numbers.

        Returns:
            Creative brief string with musical interpretation and choreographic guidance
        """
        audio = self.tools.audio_analysis
        target_duration = self.tools.get_audio_duration()
        bpm = self.tools.get_audio_bpm()
        energy = self.tools.get_audio_energy()
        is_vocal = self.tools.is_vocal_content()
        danceability = audio.get('danceability', 0.5)

        # Interpret BPM
        if bpm < 80:
            tempo_desc = "very slow, meditative tempo"
            tempo_guidance = "Use long holds (2-3 cycles), sustained emotional states, minimal transitions"
        elif bpm < 100:
            tempo_desc = "slow, contemplative tempo"
            tempo_guidance = "Focus on narrative arcs, extended expressions, storytelling structure"
        elif bpm < 130:
            tempo_desc = "moderate, conversational tempo"
            tempo_guidance = "Balance variety with coherence, natural pacing, expressive range"
        elif bpm < 150:
            tempo_desc = "fast, energetic tempo"
            tempo_guidance = "Favor short moves, rapid variety, rhythmic precision, 1-2 cycles"
        else:
            tempo_desc = "very fast, intense tempo"
            tempo_guidance = "Maximum variety, all short dance moves, single cycles, beat synchronization"

        # Interpret energy
        if energy < 0.4:
            energy_desc = "low energy, subtle dynamics"
            energy_guidance = "Minimal movement, contemplative character, gentle precision"
            suggested_emotions = "serenity1, calming1, thoughtful1, tired1"
        elif energy < 0.7:
            energy_desc = "moderate energy, balanced dynamics"
            energy_guidance = "Expressive variety, natural character, emotional range"
            suggested_emotions = "attentive1, curious1, inquiring1, amazed1, cheerful1"
        else:
            energy_desc = "high energy, vigorous dynamics"
            energy_guidance = "Maximum expression, dynamic character, bold movements"
            suggested_emotions = "enthusiastic1, electric1, laughing1, success1, cheerful1"

        # Interpret content type
        if is_vocal:
            content_desc = "vocal/lyrical content"
            content_guidance = "EMOTION moves - tell a story, express character, follow lyrical themes"
            move_type_rec = "emotion"
        else:
            content_desc = "instrumental content"
            content_guidance = "DANCE moves - synchronize to beat, emphasize rhythm, mechanical precision"
            move_type_rec = "dance"

        # Interpret danceability
        if danceability > 0.7:
            dance_desc = "highly danceable"
            dance_guidance = "Strong rhythmic focus, tight beat sync, favor dance moves even if vocal"
        elif danceability < 0.4:
            dance_desc = "low danceability"
            dance_guidance = "Looser interpretation, emotional expression prioritized over rhythm"
        else:
            dance_desc = "moderate danceability"
            dance_guidance = "Blend rhythmic and expressive elements"

        # Generate creative brief
        brief = f"""═══════════════════════════════════════════════════════════════════════════════
CREATIVE BRIEF
═══════════════════════════════════════════════════════════════════════════════

MUSICAL CHARACTER:
  Tempo: {bpm:.1f} BPM - {tempo_desc}
  Energy: {energy:.2f}/1.0 - {energy_desc}
  Content: {content_desc} ({dance_desc})
  Duration: {target_duration:.1f}s

ARTISTIC INTERPRETATION:
  This track calls for {content_guidance.lower()}

  The {tempo_desc} suggests {tempo_guidance.lower()}

  With {energy_desc}, you should aim for {energy_guidance.lower()}

CHOREOGRAPHIC DIRECTION:
  Primary Move Type: {move_type_rec.upper()}
  Pacing Strategy: {tempo_guidance}
  Character Focus: {energy_guidance}
  Rhythmic Approach: {dance_guidance}

{"SUGGESTED EMOTION PALETTE: " + suggested_emotions if is_vocal else "SUGGESTED DANCE APPROACH: Use variety of short dance moves, emphasize rhythm"}

═══════════════════════════════════════════════════════════════════════════════
YOUR CREATIVE TASK:
═══════════════════════════════════════════════════════════════════════════════

Design a choreography that captures the ESSENCE of this music, not just the duration.

STEP 1: Consider the artistic direction above
STEP 2: Use solve_duration_constraint() to get mathematically valid options
STEP 3: Evaluate solutions for artistic merit (energy match, character, flow)
STEP 4: Submit your creative choice

Remember: The solver handles math. You handle art. Make this expressive."""

        return brief
