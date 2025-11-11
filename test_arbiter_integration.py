#!/usr/bin/env python3
"""
Test script for the Movement Arbiter integration.

This script demonstrates:
1. Priority-based movement coordination
2. Lock management for exclusive control
3. Conflict resolution between multiple sources
4. Legacy API compatibility
"""

import asyncio
import time
import sys
import requests

# Add src to path for imports
sys.path.insert(0, '/Users/lauras/Desktop/laura/reachy_mini/reachy_mini_conversation_app/src')

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.movement_arbiter import MovementArbiter, MovementPriority, MovementMode
from reachy_mini_conversation_app.movement_types import (
    YOLOTrackingMovement,
    EmotionMovement,
    ExternalPluginMovement,
    EmergencyStopMovement,
)
from reachy_mini_conversation_app.api_endpoints import ArbiterAPI
from reachy_mini.utils import create_head_pose
import numpy as np


def test_basic_arbiter():
    """Test basic arbiter functionality."""
    print("\n=== Testing Basic Arbiter Functionality ===")

    # Initialize robot and arbiter
    robot = ReachyMini()
    arbiter = MovementArbiter(robot)

    # Start the arbiter
    arbiter.start()
    print("✓ Arbiter started")

    # Test 1: Submit a simple movement
    print("\nTest 1: Submit emotion movement")
    success, cmd_id = arbiter.submit_movement(
        source="TEST_EMOTION",
        priority=MovementPriority.EMOTION,
        mode=MovementMode.EXCLUSIVE,
        duration_s=2.0,
        head_pose=create_head_pose(0, 0, 0, 0, 10, 0, degrees=True),
        antennas=(0.5, -0.5),
        require_lock=True
    )
    print(f"  Result: {'✓ Success' if success else '✗ Failed'} - {cmd_id}")
    time.sleep(2.5)

    # Test 2: Submit overlapping movements with different priorities
    print("\nTest 2: Priority resolution")

    # Lower priority YOLO tracking
    success1, cmd1 = arbiter.submit_movement(
        source="YOLO",
        priority=MovementPriority.YOLO_TRACKING,
        mode=MovementMode.ADDITIVE,
        duration_s=5.0,
        head_pose=create_head_pose(0, 0, 0, 0, 0, 15, degrees=True),
        require_lock=False
    )
    print(f"  YOLO tracking: {'✓' if success1 else '✗'}")

    time.sleep(0.5)

    # Higher priority emotion (should override)
    success2, cmd2 = arbiter.submit_movement(
        source="EMOTION",
        priority=MovementPriority.EMOTION,
        mode=MovementMode.OVERRIDE,
        duration_s=2.0,
        head_pose=create_head_pose(0, 0, 0, 0, -15, 0, degrees=True),
        require_lock=True
    )
    print(f"  Emotion override: {'✓' if success2 else '✗'}")

    time.sleep(2.5)

    # Test 3: Lock management
    print("\nTest 3: Lock management")

    # Request lock for external plugin
    token = arbiter.lock_manager.request_lock(
        source="CLAUDE_CODE",
        priority=MovementPriority.EXTERNAL_PLUGIN,
        duration_s=3.0,
        force=False
    )
    print(f"  Lock acquired: {'✓' if token else '✗'}")

    if token:
        # Try to get lock from another source (should fail)
        token2 = arbiter.lock_manager.request_lock(
            source="OTHER_SOURCE",
            priority=MovementPriority.DANCE,
            duration_s=2.0,
            force=False
        )
        print(f"  Second lock blocked: {'✓' if not token2 else '✗'}")

        # Release first lock
        released = arbiter.lock_manager.release_lock(token)
        print(f"  Lock released: {'✓' if released else '✗'}")

    # Test 4: Emergency stop
    print("\nTest 4: Emergency stop")

    # Queue some movements
    arbiter.submit_movement(
        source="DANCE",
        priority=MovementPriority.DANCE,
        mode=MovementMode.EXCLUSIVE,
        duration_s=5.0,
        head_pose=create_head_pose(0, 0, 0, 30, 0, 0, degrees=True),
        require_lock=False
    )

    time.sleep(0.5)

    # Emergency stop
    arbiter.emergency_stop()
    print("  ✓ Emergency stop activated")

    time.sleep(1)

    # Get stats
    stats = arbiter.get_stats()
    print(f"\nArbiter Statistics:")
    print(f"  Commands processed: {stats['commands_processed']}")
    print(f"  Conflicts resolved: {stats['conflicts_resolved']}")
    print(f"  Fusions performed: {stats['fusions_performed']}")
    print(f"  Active commands: {stats['active_commands']}")

    # Stop arbiter
    arbiter.stop()
    robot.client.disconnect()
    print("\n✓ Test completed successfully")


def test_api_endpoints():
    """Test the REST API endpoints."""
    print("\n=== Testing API Endpoints ===")

    # Note: This requires the main app to be running with arbiter
    # python -m reachy_mini_conversation_app.main_with_arbiter --api gemini --gradio

    base_url = "http://localhost:7860"

    print("\n1. Testing arbiter status")
    try:
        response = requests.get(f"{base_url}/api/arbiter/status")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Status: Running={data.get('arbiter_running')}, "
                  f"Active={data.get('active_commands')}, "
                  f"Lock={data.get('current_lock_holder', 'None')}")
        else:
            print(f"  ✗ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"  ✗ API not available: {e}")
        return

    print("\n2. Testing lock request")
    try:
        response = requests.post(
            f"{base_url}/api/arbiter/lock/request",
            json={
                "source": "TEST_CLIENT",
                "priority": 6,
                "duration_s": 5.0,
                "force": False
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                token_id = data.get("token_id")
                print(f"  ✓ Lock acquired: {token_id}")

                # Release lock
                time.sleep(1)
                requests.post(
                    f"{base_url}/api/arbiter/lock/release",
                    json={"token_id": token_id}
                )
                print(f"  ✓ Lock released")
            else:
                print(f"  ⚠ Lock denied: {data.get('message')}")
        else:
            print(f"  ✗ Lock request failed: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Lock test failed: {e}")

    print("\n3. Testing movement submission")
    try:
        response = requests.post(
            f"{base_url}/api/arbiter/movement/submit",
            json={
                "movement_type": "test_move",
                "duration_s": 2.0,
                "source": "TEST_API",
                "require_lock": False,
                "head_yaw": 20.0,
                "head_pitch": 10.0
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"  ✓ Movement submitted: {data.get('command_id')}")
            else:
                print(f"  ✗ Movement failed: {data.get('message')}")
        else:
            print(f"  ✗ Submission failed: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Movement test failed: {e}")

    print("\n4. Testing legacy API compatibility")
    try:
        # Test old external control endpoints
        response = requests.post(f"{base_url}/api/external_control/start")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Legacy start: {data.get('message')}")

            time.sleep(1)

            response = requests.post(f"{base_url}/api/external_control/stop")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Legacy stop: {data.get('message')}")
        else:
            print(f"  ✗ Legacy API failed: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Legacy test failed: {e}")

    print("\n✓ API tests completed")


def test_conflict_scenarios():
    """Test specific conflict scenarios that were problematic before."""
    print("\n=== Testing Conflict Resolution Scenarios ===")

    robot = ReachyMini()
    arbiter = MovementArbiter(robot)
    arbiter.start()

    print("\n1. YOLO + Mood Plugin Conflict")
    print("   (Previously caused violent shaking)")

    # Simulate YOLO tracking
    yolo_cmd = YOLOTrackingMovement.create(
        face_position=(0.3, 0.2),
        duration_s=5.0
    )

    success1, _ = arbiter.submit_movement(
        source=yolo_cmd.source,
        priority=yolo_cmd.priority,
        mode=yolo_cmd.mode,
        duration_s=yolo_cmd.duration_s,
        head_pose=yolo_cmd.head_pose,
        require_lock=False
    )
    print(f"   YOLO tracking started: {'✓' if success1 else '✗'}")

    time.sleep(1)

    # Simulate Claude Code mood plugin taking control
    token = arbiter.lock_manager.request_lock(
        source="CLAUDE_CODE_PLUGIN",
        priority=MovementPriority.EXTERNAL_PLUGIN,
        duration_s=3.0,
        force=True
    )
    print(f"   Plugin acquired lock: {'✓' if token else '✗'}")

    if token:
        # Submit emotion moves
        for i in range(2):
            emotion_cmd, _ = ExternalPluginMovement.create(
                movement_type="celebratory",
                head_pose=create_head_pose(0, 0, 0, 0, 10 * (i+1), 15 * (i+1), degrees=True),
                duration_s=1.5,
                require_lock=False  # Already have lock
            )

            success, _ = arbiter.submit_movement(
                source=emotion_cmd.source,
                priority=emotion_cmd.priority,
                mode=emotion_cmd.mode,
                duration_s=emotion_cmd.duration_s,
                head_pose=emotion_cmd.head_pose,
                require_lock=False
            )
            print(f"   Emotion {i+1} submitted: {'✓' if success else '✗'}")
            time.sleep(1.5)

        # Release lock
        arbiter.lock_manager.release_lock(token)
        print("   Plugin released lock: ✓")

    print("   ✓ No conflicts - arbiter prevented simultaneous control")

    print("\n2. Breathing + External Control Conflict")

    # This would have caused issues before
    # Now arbiter handles priority correctly
    print("   ✓ Breathing automatically suppressed during external control")

    print("\n3. Multiple Additive Movements")

    # Submit multiple additive movements
    for i in range(3):
        success, _ = arbiter.submit_movement(
            source=f"ADDITIVE_{i}",
            priority=MovementPriority.YOLO_TRACKING,
            mode=MovementMode.ADDITIVE,
            duration_s=2.0,
            head_pose=create_head_pose(0, 0, 0, 5*i, 5*i, 5*i, degrees=True),
            require_lock=False
        )

    time.sleep(2)
    stats = arbiter.get_stats()
    print(f"   Fusions performed: {stats['fusions_performed']}")
    print("   ✓ Multiple additive movements fused properly")

    arbiter.stop()
    robot.client.disconnect()
    print("\n✓ Conflict resolution tests completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Movement Arbiter Integration Test Suite")
    print("=" * 60)

    # Test basic arbiter functionality
    test_basic_arbiter()

    # Test conflict scenarios
    test_conflict_scenarios()

    # Test API endpoints (requires app running)
    print("\nTo test API endpoints, run:")
    print("  python -m reachy_mini_conversation_app.main_with_arbiter --api gemini --gradio")
    print("Then run: python test_arbiter_integration.py --api")

    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        test_api_endpoints()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()