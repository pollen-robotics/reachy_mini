# Changelog - Reachy Mini Choreography Builder

All notable changes to the Choreography Builder interface are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2025-10-18

### 🎉 Initial Release

Complete choreography builder interface for Reachy Mini beta testing program.

---

## Features Implemented

### 🎮 Manual Position Control

**Added:**
- 6-DOF manual control interface
  - X, Y, Z position sliders (-40mm to 40mm range)
  - Yaw, Pitch, Roll rotation sliders (-60° to 60° range)
  - Left/Right antenna sliders (-3 to 3 range)
- Dual input modes for all axes
  - Range sliders for quick adjustment (0.2 step increments)
  - Number inputs for precise values (0.01 precision)
- Two-way synchronization between sliders and number inputs
- Visual safety zones using color gradients
  - Blue zones indicate safe operating range
  - Pink zones warn of extreme positions
- Configurable movement duration (0.1-5.0 seconds)
- Degrees/radians toggle for rotation values
- "Execute" button sends position commands to daemon
- Real-time status feedback with color-coded messages

**Slider Gradients:**
- Position sliders (X, Y, Z): Blue center to pink extremes
- Rotation sliders: Pink extremes, blue center (symmetric)
- Antenna sliders: Blue safe zone (-3 to 1) to pink danger zone (1 to 3)
- Right antenna: Visually reversed using CSS scaleX(-1) transform

### 🔗 Binding Mechanisms

**Added:**
- **Yaw Binding** (`bindYaw` checkbox)
  - Synchronizes left and right yaw values
  - Delta-based: maintains relative offset between sides
  - Useful for coordinated head turning

- **Antenna Normal Binding** (`bindAnt` checkbox)
  - Both antennae move to same value
  - Creates symmetrical antenna poses

- **Antenna Inverse Binding** (`invAnt` checkbox)
  - Antennae mirror each other (opposite values)
  - Creates expressive mirrored gestures

- **Mutual Exclusion Logic**
  - Normal and inverse antenna binding are mutually exclusive
  - Checking one automatically unchecks the other
  - Prevents conflicting binding modes

### 🎭 Pre-Recorded Move Library

**Added:**
- Dynamic move loading from `moves.json` file
  - 20 dance moves from `pollen-robotics/reachy-mini-dances-library`
  - 81 emotion moves from `pollen-robotics/reachy-mini-emotions-library`
  - Automatic alphabetical sorting
- Radio button selection interface
  - Two-column layout (dances left, emotions right)
  - Visual separation with distinct section colors
  - Move counts displayed in headers
- "Execute Move" button (light blue)
  - Sends API request to daemon
  - Returns UUID for tracking
  - Displays execution status
- "Stop" button (pink)
  - Cancels currently running move
  - Uses UUID to target specific move
- Status indicator with color-coded feedback
  - Green: Success, move executing
  - Pink: Error, move failed
  - Yellow: Processing, waiting

### 🎵 Choreography Builder

**Added:**
- Routine sequence builder
  - Add selected moves to ordered list
  - Display shows: move name, cycles, amplitude
  - Move counter shows total routine length
  - Clear button removes all moves (pink, destructive action)
- Per-move configuration
  - **Cycles**: Number of repetitions (1-10 range, default 4)
  - **Amplitude**: Movement intensity (0.1-2.0 range, default 1.0)
- BPM (Beats Per Minute) configuration
  - Range: 40-200 BPM
  - Default: 120 BPM
  - Affects choreography timing
- "Add to Routine" button (yellow)
  - Appends configured move to sequence
  - Updates routine display immediately
- JSON Export functionality
  - "💾 Export JSON" button (yellow)
  - Generates official Reachy Mini choreography format
  - Downloads as `choreography_{bpm}bpm.json`
  - Includes all moves with cycles and amplitude
  - Ready to play via Python SDK or REST API

**Choreography JSON Format:**
```json
{
    "bpm": 120,
    "sequence": [
        {
            "move": "move_name",
            "cycles": 4,
            "amplitude": 1.0
        }
    ]
}
```

### 📹 Live Monitoring

**Added:**
- **3D Simulator Video Feed**
  - MJPEG stream from MuJoCo simulator
  - 500px tall for better visibility
  - Auto-connects to daemon camera endpoint
  - Connection status indicator (green/pink)
  - Error handling for disconnection

- **Live Head Pose Chart** (Chart.js)
  - Real-time bar chart showing 8 axes:
    - X, Y, Z position (mm)
    - Yaw, Pitch, Roll rotation (degrees)
    - Left & Right antenna position
  - Color-coded bars using Pollen Robotics palette:
    - X: Pink, Y: Yellow, Z: Light Blue
    - Yaw: Light Green, Pitch: Pink, Roll: Yellow
    - Ant L: Light Blue, Ant R: Light Green
  - WebSocket-powered (ws://localhost:8100/api/state/stream)
  - Updates ~10 times per second
  - Connection status indicator
  - Minimal animation for smooth real-time updates

### 👁️ Quick Look Presets

**Added:**
- 9-button directional pad
  - 8 directional buttons (up, down, left, right, diagonals)
  - Center button for reset to neutral
  - Color-coded buttons (green, blue, yellow, pink distributed)
  - Instant execution (0.5s duration)
- Quick preset buttons
  - "Look Up" (light green)
  - "Look Down" (light blue)
  - "Look Left" (yellow)
  - "Look Right" (pink)
  - "Reset" (navy)
- All presets use optimized values for natural-looking movements

### 🎨 Pollen Robotics Branding

**Added:**
- Official color palette from Pollen Robotics logo
  - Navy Blue (#2B4C7E): Primary brand color
  - Light Blue (#3bb0d1): Safe zones, info
  - Light Green (#3dde99): Success, connected
  - Yellow Gold (#ffc261): Highlights, accents
  - Pink (#ff6170): Danger zones, errors
  - White (#FFFFFF): Text, backgrounds
- CSS Custom Properties for easy theming
- Distributed color usage throughout interface
  - Section borders: Different colors per section
  - Buttons: Individual color classes (btn-navy, btn-yellow, btn-blue, btn-pink, btn-green)
  - Status messages: Color-coded by type
  - Chart bars: Unique color per axis
  - Slider gradients: Blue (safe) to pink (danger)
- Gradient effects on all buttons with hover states
- Consistent visual language across all components

### 📊 Layout & UX

**Added:**
- Two-column responsive layout
  - Left: Monitoring (video, chart, quick controls)
  - Right: Control panels (manual, moves, choreography)
- Optimized space utilization
  - Video feed: 500px tall (increased from 330px)
  - Look pad and chart: Side-by-side in 2-column grid
  - Compact slider rows for all controls
- Visual hierarchy
  - Section headers (h2) in navy blue
  - Subsection headers (h3) with emoji icons
  - Clear visual separation between sections
- Consistent spacing and padding
  - 16px padding on sections
  - 12px gap between elements
  - 3px borders for emphasis
- Shadow effects for depth
  - Sections: 4px blur, subtle shadow
  - Buttons: 2-6px blur depending on prominence
- Smooth transitions on hover states

### 🔧 Technical Implementation

**Added:**
- Single-file architecture (HTML + CSS + JavaScript)
  - No build step required
  - Easy distribution and versioning
  - Works with file:// protocol (HTTP server recommended)
- Pure vanilla JavaScript (ES6+)
  - No framework dependencies
  - Lightweight and fast
  - Easy to understand and modify
- External dependencies
  - Chart.js 3.9.1 (CDN) for visualization
- REST API integration
  - Manual control: POST /api/joints/target
  - Move execution: POST /api/move/play/recorded-move-dataset/{dataset}/{move}
  - Move stop: POST /api/move/stop
  - Video stream: GET /api/camera/stream.mjpg
- WebSocket integration
  - Real-time state stream: ws://localhost:8100/api/state/stream
  - JSON message parsing
  - Automatic chart updates
- Blob API for JSON export
  - Client-side file generation
  - No server processing needed
  - Standard browser download UX
- Error handling throughout
  - Try-catch blocks on all async operations
  - User-friendly error messages
  - Console logging for debugging
  - Fallback values for missing data

### 🎯 User Experience

**Added:**
- Tooltips on slider labels
  - Hover over labels for helpful descriptions
  - Range information included
- Color-coded status messages
  - Green: Success, connected, ready
  - Pink: Error, disconnected, warning
  - Yellow: Processing, info, neutral
- Auto-clearing status messages
  - Temporary messages clear after 5 seconds
  - Persistent states remain (connected/disconnected)
- Disabled states for invalid operations
  - Clear routine button disabled when empty
  - Export disabled when no moves
- Input validation and clamping
  - All number inputs clamped to valid ranges
  - Step increments enforced
  - Invalid values auto-corrected
- Alphabetical sorting of moves
  - Easier to find specific moves
  - Consistent ordering across sessions

---

## Design Decisions

### Why Single-File Architecture?

**Rationale:**
- Zero dependencies (beyond CDN Chart.js)
- No build step or tooling required
- Easy to share with beta testers
- Simple version control (one file)
- Works offline (except Chart.js CDN)
- Accessible to beginners

**Tradeoffs:**
- Larger file size (~1700 lines)
- No code splitting
- Global namespace pollution (mitigated with const/let)

**Verdict:** Benefits outweigh costs for this use case.

### Why Vanilla JavaScript Over Framework?

**Rationale:**
- Small codebase (~800 lines JS)
- No complex state management needed
- Excellent performance without framework overhead
- Lower barrier to entry for contributors
- No build tooling complexity

**Tradeoffs:**
- Manual DOM manipulation
- More verbose than React/Vue
- No virtual DOM optimizations

**Verdict:** Vanilla JS is optimal for this tool.

### Why CSS Custom Properties?

**Rationale:**
- Centralized color management
- Easy theme customization
- No CSS preprocessor needed
- Excellent browser support
- Runtime color changes possible

**Benefits:**
- Change entire theme by editing 6 variables
- Consistent color usage throughout
- Maintainable and readable

### Why scaleX(-1) for Antenna Reversal?

**Rationale:**
- Pure CSS solution (no JavaScript complexity)
- Automatically flips gradients correctly
- Visual-only transformation (values unchanged)
- Eliminates need for value negation logic

**Initial Mistake:**
- Attempted to negate values in JavaScript
- Created confusion between visual and actual values
- Led to complex binding logic

**Solution:**
- Use scaleX(-1) for visual flip only
- Both sliders have same min/max/value ranges
- Gradients are identical (flip happens automatically)
- JavaScript logic simplified significantly

**Lesson:** Use CSS transforms for visual changes, not JavaScript.

### Why Chart.js Over Custom Canvas?

**Rationale:**
- Proven, well-tested library
- Beautiful default styling
- Easy bar chart configuration
- Built-in responsive behavior
- Active maintenance and community

**Tradeoffs:**
- External dependency (CDN)
- Slight overhead (~50kb)
- Less customization than raw canvas

**Verdict:** Chart.js is the right choice for maintainability.

### Why Blob API for Export?

**Rationale:**
- No server-side processing needed
- Works entirely in browser
- Standard download UX
- Small, clean implementation

**Alternative Considered:**
- Data URI approach (works but less clean)
- Server endpoint (unnecessary complexity)

**Verdict:** Blob API is modern, clean solution.

### Why Mutual Exclusion for Antenna Bindings?

**Rationale:**
- Normal and inverse binding are contradictory
- Both enabled simultaneously would conflict
- Creates clear user expectations

**Implementation:**
- Event listeners on both checkboxes
- Checking one unchecks the other
- Both can be unchecked (independent control)

**UX Benefit:** Prevents confusion and undefined behavior.

### Why 67% Gradient Blend for Antennas?

**Rationale:**
- Antenna range: -3 to 3 (6 unit span)
- Safe zone: -3 to 1 (4 units)
- Danger zone: 1 to 3 (2 units)
- Math: 4 / 6 = 0.6667 = 67%

**Result:**
- Blend point visually centered over value 1
- Clear indication of danger zone
- Sharper transition than 50% blend

### Why 500px Video Height?

**Rationale:**
- Previous 330px was too small
- Wasted vertical space in left column
- 500px provides better view of simulator
- Still fits on standard laptop screens

**Tradeoff:**
- Less vertical space for chart below
- Acceptable given side-by-side layout change

---

## Technical Challenges & Solutions

### Challenge 1: Antenna Slider Visual Reversal

**Problem:**
- Right antenna needed to move opposite direction visually
- Initial attempt negated values in JavaScript
- Created confusion between visual position and actual value
- Gradient colors were backwards

**Solution:**
- Use CSS `scaleX(-1)` for visual flip only
- No JavaScript value negation
- Both sliders use identical gradient
- Transform automatically flips gradient correctly

**Code:**
```css
.reversed-slider {
    transform: scaleX(-1);
}
```

**Result:** Clean, simple, correct behavior.

### Challenge 2: Dynamic Move Loading

**Problem:**
- 101 moves in HTML was unmaintainable
- Adding/removing moves required HTML editing
- No version control for move list

**Solution:**
- Externalize moves to `moves.json`
- Fetch JSON on page load
- Dynamically generate radio buttons with JavaScript
- Automatic alphabetical sorting

**Benefits:**
- Add moves by editing one JSON file
- No HTML changes needed
- Easy to version control
- Alphabetical sorting automatic

### Challenge 3: Color Palette Distribution

**Problem:**
- Initial design used too much of one color
- Not utilizing full Pollen Robotics palette
- Lack of visual variety

**Solution:**
- Create individual button color classes
- Distribute colors evenly across interface
- Use semantic color coding (green=success, pink=error)
- Chart bars each get unique color

**Result:** Vibrant, balanced, branded interface.

### Challenge 4: Binding Logic Complexity

**Problem:**
- Yaw binding, antenna normal binding, antenna inverse binding
- All interact with same input handlers
- Risk of infinite loops (A changes B, B changes A)

**Solution:**
- Single event handler per input
- Check all binding states sequentially
- Update paired inputs directly (no events fired)
- Mutual exclusion prevents conflicting bindings

**Lesson:** Centralized binding logic in input handlers prevents issues.

### Challenge 5: WebSocket Reconnection

**Current State:**
- No automatic reconnection implemented
- User must refresh page

**Reason:**
- Keeps code simple for v1.0
- Daemon restarts are rare in practice
- Manual refresh is acceptable workaround

**Future Enhancement:**
```javascript
function connectWebSocket() {
    ws = new WebSocket(url);
    ws.onclose = () => {
        setTimeout(connectWebSocket, 3000);
    };
}
```

### Challenge 6: Chart Performance

**Problem:**
- WebSocket sends ~10 messages/second
- Chart.js animation could cause lag
- Need smooth real-time updates

**Solution:**
- Set animation duration to 100ms (minimal)
- Use `chart.update('none')` for instant updates
- Browser throttles at 60 FPS naturally

**Result:** Smooth, responsive chart updates.

---

## Known Limitations

### Current Version (1.0.0)

1. **No Mobile Responsiveness**
   - Designed for desktop browsers
   - Layout breaks on small screens
   - Touch controls not optimized

2. **No WebSocket Reconnection**
   - Disconnect requires page refresh
   - No automatic retry logic

3. **No Undo/Redo for Choreography**
   - Removing moves from routine not supported
   - Must clear and rebuild

4. **No Move Preview**
   - Can't see what move looks like before executing
   - Move names are descriptive but not visual

5. **No Choreography Import**
   - Export works, but no import function
   - Can't load previously exported choreographies

6. **No Preset Saving**
   - Manual positions can't be saved
   - Must re-enter values each session

7. **No Keyboard Shortcuts**
   - All actions require clicking
   - Power users can't use hotkeys

8. **No Move Search/Filter**
   - 81 emotions require scrolling
   - No search box to find moves quickly

9. **No Multi-Select for Moves**
   - Can't select multiple moves at once
   - Must add to routine one at a time

10. **No Live Preview in Choreography**
    - Can't preview choreography before export
    - Must export, then test separately

---

## Future Roadmap

### Version 1.1 (Planned)

**UX Improvements:**
- [ ] Add keyboard shortcuts (Space = execute, Esc = stop)
- [ ] Add move search/filter functionality
- [ ] Add undo/redo for choreography builder
- [ ] Add routine move reordering (drag-and-drop)
- [ ] Add preset saving/loading for manual positions
- [ ] Add move descriptions on hover

**Technical Improvements:**
- [ ] Implement WebSocket auto-reconnection
- [ ] Add debouncing to slider inputs
- [ ] Add choreography import functionality
- [ ] Add error retry logic
- [ ] Add offline mode detection

### Version 2.0 (Future)

**Major Features:**
- [ ] Mobile responsive layout
- [ ] Touch gesture support
- [ ] Choreography timeline view
- [ ] Move preview on hover
- [ ] Multi-select for batch operations
- [ ] Custom move recording from UI
- [ ] Choreography sharing (upload/download)
- [ ] User accounts and cloud storage

**Technical Upgrades:**
- [ ] TypeScript migration
- [ ] Module bundling (webpack/vite)
- [ ] Automated testing (Jest)
- [ ] Performance monitoring
- [ ] Accessibility audit and fixes
- [ ] PWA capabilities

---

## Changelog Format

### Types of Changes

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

---

## Version History

### [1.0.0] - 2025-10-18

**Initial release for Pollen Robotics beta testing program.**

All features listed above in "Features Implemented" section.

**Contributors:**
- Carson (LAURA Project) - Interface design and implementation
- Pollen Robotics Team - Reachy Mini SDK and move libraries
- Hugging Face Team - Move library hosting

**Tested With:**
- Reachy Mini daemon v1.0.0+
- MuJoCo simulator 3.0+
- Chrome 90+, Firefox 88+, Safari 14+
- macOS 14+ (primary platform)

---

## Migration Guide

N/A - This is the initial release.

Future versions will include migration guides if breaking changes occur.

---

## Deprecation Policy

This project follows semantic versioning:
- **Major version (X.0.0):** Breaking changes, API changes
- **Minor version (1.X.0):** New features, backwards compatible
- **Patch version (1.0.X):** Bug fixes, backwards compatible

**Deprecation Process:**
1. Feature marked as deprecated in CHANGELOG
2. Warning added to UI/console
3. Minimum one minor version before removal
4. Removal in next major version

---

## Support & Feedback

**Bug Reports:**
- GitHub Issues: https://github.com/pollen-robotics/reachy_mini/issues
- Include: Browser version, OS, daemon version, steps to reproduce

**Feature Requests:**
- GitHub Discussions: https://github.com/pollen-robotics/reachy_mini/discussions
- Describe use case and expected behavior

**Community:**
- Pollen Robotics Forum: https://forum.pollen-robotics.com
- Discord: https://discord.gg/pollen-robotics

---

**Last Updated:** October 18, 2025
**Maintainer:** Carson (LAURA Project)
**License:** Apache 2.0 (aligns with Reachy Mini SDK)
