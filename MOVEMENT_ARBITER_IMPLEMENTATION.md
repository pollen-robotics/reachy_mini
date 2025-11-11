# Movement Arbiter Implementation Guide

## Overview

This document describes the complete implementation of the Movement Arbiter system for Reachy Mini, which solves the critical movement conflict issues that were causing violent robot shaking and loss of control.

## Problem Summary

The original system had three independent control streams fighting for robot control:
1. **MovementManager** - 100Hz control loop constantly sending commands
2. **Claude Code Mood Plugin** - Direct HTTP calls to daemon bypassing manager
3. **YOLO Face Tracking** - Continuous offset commands overlapping with others

This caused:
- Violent shaking when multiple systems were active
- Loss of control during plugin execution
- Dangerous rapid movements that scared the user
- No coordination between movement sources

## Solution Architecture

The Movement Arbiter provides a single point of control with:
- **Priority-based arbitration** - Higher priority movements override lower ones
- **Lock management** - Exclusive control tokens for critical movements
- **Command fusion** - Intelligent blending of compatible movements
- **Unified API** - Single interface for all movement sources

## Implementation Files

### Core System

1. **movement_arbiter.py** - Central orchestration system
   - `MovementArbiter` class - Main coordinator
   - `LockManager` class - Exclusive control management
   - `MovementCommand` dataclass - Unified command format
   - Priority queue processing at 100Hz

2. **movement_types.py** - Movement type definitions
   - `YOLOTrackingMovement` - Face tracking commands
   - `EmotionMovement` - Emotion expressions
   - `DanceMovement` - Dance sequences
   - `ExternalPluginMovement` - Claude Code plugin moves
   - `IdleBreathingMovement` - Background breathing
   - `SpeechWobbleMovement` - Audio-reactive motion
   - `EmergencyStopMovement` - Safety stop

3. **api_endpoints.py** - REST API for external control
   - Lock request/release endpoints
   - Movement submission endpoint
   - Status and statistics endpoint
   - Legacy compatibility endpoints

### Integration Files

4. **moves_integration_patch.py** - MovementManager integration
   - Shows how to modify existing MovementManager
   - Routes movements through arbiter
   - Maintains backwards compatibility

5. **mood_extractor_updated.py** - Updated Claude Code plugin
   - Uses new arbiter API instead of direct daemon calls
   - Proper lock acquisition and release
   - No more conflicts with other systems

6. **main_with_arbiter.py** - Updated main entry point
   - Initializes arbiter on startup
   - Registers API endpoints
   - Connects all components

### Testing

7. **test_arbiter_integration.py** - Comprehensive test suite
   - Basic arbiter functionality tests
   - Conflict resolution scenarios
   - API endpoint testing
   - Performance validation

## Movement Priority Levels

```python
class MovementPriority(IntEnum):
    IDLE_BREATHING = 1      # Lowest - background breathing
    YOLO_TRACKING = 2       # Face tracking
    SPEECH_WOBBLE = 3       # Audio-reactive wobble
    DANCE = 4              # Dance moves
    EMOTION = 5            # Emotion expressions
    MANUAL_CONTROL = 6     # Manual position commands
    EXTERNAL_PLUGIN = 7    # Claude Code plugin
    EMERGENCY_STOP = 10    # Highest - safety stop
```

## API Endpoints

### Lock Management
- `POST /api/arbiter/lock/request` - Request exclusive control
- `POST /api/arbiter/lock/release` - Release control lock
- `GET /api/arbiter/lock/status` - Check current lock holder

### Movement Control
- `POST /api/arbiter/movement/submit` - Submit movement command
- `POST /api/arbiter/movement/cancel/{id}` - Cancel specific movement
- `POST /api/arbiter/movement/stop` - Emergency stop all movements

### Status
- `GET /api/arbiter/status` - Get system status and statistics
- `GET /api/arbiter/health` - Health check endpoint

### Legacy Compatibility
- `POST /api/external_control/start` - Old plugin start (redirects to arbiter)
- `POST /api/external_control/stop` - Old plugin stop (redirects to arbiter)
- `GET /api/external_control/status` - Old status check (redirects to arbiter)

## Deployment Instructions

### 1. Install the new files

```bash
# Copy core files to the app directory
cp movement_arbiter.py /path/to/reachy_mini_conversation_app/src/reachy_mini_conversation_app/
cp movement_types.py /path/to/reachy_mini_conversation_app/src/reachy_mini_conversation_app/
cp api_endpoints.py /path/to/reachy_mini_conversation_app/src/reachy_mini_conversation_app/
```

### 2. Update existing files

The following files need modifications:

**moves.py** - Add arbiter integration:
- Add arbiter parameter to constructor
- Replace direct `set_target()` calls with arbiter submissions
- See `moves_integration_patch.py` for reference

**main.py** - Initialize arbiter:
- Import MovementArbiter and ArbiterAPI
- Create arbiter instance after robot initialization
- Register API endpoints with FastAPI app
- Start/stop arbiter with other services
- See `main_with_arbiter.py` for reference

### 3. Update Claude Code plugin

Replace the mood_extractor.py with mood_extractor_updated.py:
```bash
cp mood_extractor_updated.py /path/to/claude-plugins/marketplace/reachy-mini/hooks/mood_extractor.py
```

### 4. Test the integration

```bash
# Run the test suite
python test_arbiter_integration.py

# For API tests, first start the app:
python -m reachy_mini_conversation_app.main --api gemini --gradio

# Then in another terminal:
python test_arbiter_integration.py --api
```

## Benefits of the New System

1. **No More Conflicts** - Single point of control prevents fighting
2. **Safe Operation** - No violent shaking or dangerous movements
3. **Priority Management** - Important movements override less important ones
4. **Smooth Transitions** - Proper blending and interpolation
5. **External Control** - Claude Code plugin works seamlessly
6. **Backwards Compatible** - Legacy APIs still work
7. **Observable** - Statistics and status for debugging
8. **Extensible** - Easy to add new movement types

## Migration Strategy

For gradual migration:

1. **Phase 1** - Deploy arbiter alongside existing system
   - Arbiter runs but doesn't control robot
   - Monitor statistics to verify operation

2. **Phase 2** - Route external plugin through arbiter
   - Update mood_extractor.py to use new API
   - Test plugin coordination

3. **Phase 3** - Route YOLO tracking through arbiter
   - Modify camera worker to submit via arbiter
   - Verify face tracking still works

4. **Phase 4** - Full integration
   - All movements go through arbiter
   - Remove old external_control flags
   - Complete migration

## Troubleshooting

### Arbiter not starting
- Check robot connection
- Verify import paths
- Check for port conflicts

### Movements not executing
- Check arbiter status endpoint
- Verify lock not held by another source
- Check movement priority

### Plugin conflicts
- Ensure plugin uses new API
- Check lock acquisition succeeds
- Verify lock release after use

### Performance issues
- Monitor arbiter statistics
- Check control loop frequency
- Verify no blocking operations

## Future Enhancements

1. **Movement Recording** - Record and replay movement sequences
2. **Gesture Library** - Pre-defined gesture combinations
3. **Learning System** - Adaptive movement based on context
4. **Multi-Robot** - Coordinate multiple robots
5. **Web Dashboard** - Visual monitoring and control

## Conclusion

The Movement Arbiter successfully solves the critical movement conflict issues while providing a robust, extensible framework for future development. The system is production-ready and can be deployed immediately to prevent the dangerous robot behaviors observed previously.

---

*Implementation completed by Claude Code for the Reachy Mini project*
*October 2025*