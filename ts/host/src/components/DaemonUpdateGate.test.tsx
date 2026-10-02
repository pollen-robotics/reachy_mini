/**
 * Behavioral tests for `DaemonUpdateGate`, the one stateful component
 * of the update flow (three timers, two refs-as-latches, phase derived
 * during render).
 *
 * Everything here runs on fake timers: the component's failure modes
 * are all about WHEN things happen (stall timeout vs progress lines,
 * reboot budget, latches surviving re-renders), so wall-clock tests
 * would either lie or take minutes.
 *
 * The last describe block pins the regressions called out in the
 * #1316 review (all fixed since - these tests are the guard rails):
 *
 *  - the stall timeout must measure SILENCE, not total install time -
 *    a slow-but-progressing install (pip on weak wifi) must not be
 *    declared dead at a hard 180 s ceiling;
 *  - after the gate locally times out and the user dismisses it, a
 *    late `rebooting` progress frame (the embed's sessionStopped
 *    translator firing on a normal end-session) must NOT resurrect
 *    the full-screen overlay nor fire `onSessionLost` mid-leave;
 *  - dismissing the notice must work without a parent re-render.
 *
 * If one of these goes red, a fix regressed - do not "fix" the test
 * by asserting the buggy behavior.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';

import {
  DaemonUpdateGate,
  type DaemonUpdateGateProps,
  type DaemonUpdateProgress,
} from './DaemonUpdateGate';

/** Mirrors `UPDATE_STALL_TIMEOUT_MS` in the component (not exported). */
const STALL_MS = 180_000;
/** Mirrors `REBOOT_TIMEOUT_MS` in the component (not exported). */
const REBOOT_MS = 240_000;

/** Below `MIN_SUPPORTED_DAEMON_VERSION` (1.8.2): hard block, no OTA. */
const ANCIENT = '1.7.0';
/** At the floor but behind latest: soft notice, OTA available. */
const BEHIND = '1.8.2';
const LATEST = '1.9.0';

/* ─────────────────── Harness ─────────────────── */

function setup(overrides: Partial<DaemonUpdateGateProps> = {}) {
  const onStartUpdate = vi.fn();
  const onCancelUpdate = vi.fn();
  const onExitApp = vi.fn();
  const onSessionLost = vi.fn();
  const onDismiss = vi.fn();
  const props: DaemonUpdateGateProps = {
    currentVersion: BEHIND,
    latestVersion: LATEST,
    progress: null,
    robotBackOnline: false,
    appLive: true,
    onStartUpdate,
    onCancelUpdate,
    onExitApp,
    onSessionLost,
    onDismiss,
    ...overrides,
  };
  const view = render(<DaemonUpdateGate {...props} />);
  /** Re-render with a partial prop patch (callbacks stay stable, the
   *  way the shell's useCallback-wrapped handlers do). */
  const update = (patch: Partial<DaemonUpdateGateProps>): void => {
    Object.assign(props, patch);
    view.rerender(<DaemonUpdateGate {...props} />);
  };
  /** Deliver one `embed:update-progress` frame. Always a fresh object:
   *  the progress effect keys on reference identity, exactly like the
   *  shell's `setUpdateProgress` producing a new payload per message. */
  const deliver = (frame: DaemonUpdateProgress): void => {
    update({ progress: { ...frame } });
  };
  return {
    update,
    deliver,
    onStartUpdate,
    onCancelUpdate,
    onExitApp,
    onSessionLost,
    onDismiss,
  };
}

/** Click through the notice card into the `updating` phase. */
function startUpdate(h: ReturnType<typeof setup>): void {
  fireEvent.click(screen.getByRole('button', { name: 'Update' }));
  expect(h.onStartUpdate).toHaveBeenCalledTimes(1);
}

function advance(ms: number): void {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

/* ─────────────────── Prompt derivation ─────────────────── */

describe('DaemonUpdateGate prompt derivation', () => {
  it('renders nothing when the daemon is up to date', () => {
    setup({ currentVersion: LATEST, latestVersion: LATEST });
    expect(document.body.textContent).toBe('');
  });

  it('renders nothing when the daemon version is unknown (fail-open)', () => {
    setup({ currentVersion: null, latestVersion: LATEST });
    expect(document.body.textContent).toBe('');
  });

  it('blocks a pre-OTA daemon immediately, without waiting for appLive', () => {
    setup({ currentVersion: ANCIENT, appLive: false });
    // Full-screen tier, and the honest copy: this daemon cannot OTA,
    // so the only offer is the desktop app - no dead "Update now".
    expect(screen.getByText('Update from the desktop app')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Update now' })).toBeNull();
    expect(screen.getByRole('link', { name: 'Get the desktop app' })).toBeTruthy();
  });

  it('the blocking tier offers an exit back to the picker', () => {
    // The block only paints after startSession() succeeded, so a live
    // session (and an awake robot) sits behind it: two outbound links
    // alone would leave closing the tab as the only release path.
    const h = setup({ currentVersion: ANCIENT, appLive: false });
    fireEvent.click(
      screen.getByRole('button', { name: 'Back to my Reachies' }),
    );
    expect(h.onExitApp).toHaveBeenCalledTimes(1);
  });

  it('holds the soft notice until the app is live', () => {
    const h = setup({ appLive: false });
    expect(document.body.textContent).toBe('');
    h.update({ appLive: true });
    expect(screen.getByText('A Reachy update is available')).toBeTruthy();
    // Card, not gate: the app behind it stays usable.
    expect(screen.getByRole('button', { name: 'Not now' })).toBeTruthy();
  });

  it('"Not now" dismisses for the rest of the session', () => {
    const h = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(h.onDismiss).toHaveBeenCalledTimes(1);
    // In situ the shell's onDismiss handler updates its own state and
    // re-renders the gate; simulate that parent re-render. (The gate
    // not clearing WITHOUT it is pinned as a regression below.)
    h.update({});
    expect(document.body.textContent).toBe('');
    // A later re-render (late version read, list refresh) must not
    // reopen a card the user already answered.
    h.update({ latestVersion: LATEST });
    expect(document.body.textContent).toBe('');
  });
});

/* ─────────────────── progress → phase ─────────────────── */

describe('DaemonUpdateGate update flow', () => {
  it('Update click flips to the full-screen updating phase', () => {
    const h = setup();
    startUpdate(h);
    expect(screen.getByText('Updating your Reachy…')).toBeTruthy();
  });

  it('daemon-reported failure lands on the failed screen with the error', () => {
    const h = setup();
    startUpdate(h);
    h.deliver({ status: 'failed', error: 'pip exploded' });
    expect(screen.getByText("Update didn't finish")).toBeTruthy();
    expect(screen.getByText(/pip exploded/)).toBeTruthy();
    expect(h.onSessionLost).not.toHaveBeenCalled();
    // The embed disarmed itself on its own `failed`: no cancel needed.
    expect(h.onCancelUpdate).not.toHaveBeenCalled();
  });

  it('rebooting fires onSessionLost exactly once, across duplicate terminal frames', () => {
    const h = setup();
    startUpdate(h);
    h.deliver({ status: 'rebooting' });
    expect(screen.getByText('Reachy is rebooting…')).toBeTruthy();
    expect(h.onSessionLost).toHaveBeenCalledTimes(1);
    // The daemon may still flush a `done` (or a re-render replay a
    // frame): the one-way door must not swing twice.
    h.deliver({ status: 'done' });
    expect(h.onSessionLost).toHaveBeenCalledTimes(1);
  });

  it('robot back online completes the gate; "Back to my Reachies" dismisses', () => {
    const h = setup();
    startUpdate(h);
    h.deliver({ status: 'rebooting' });
    h.update({ robotBackOnline: true });
    expect(screen.getByText('Reachy is up to date')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Back to my Reachies' }));
    expect(h.onDismiss).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toBe('');
  });

  it('a robot that never comes back trips the reboot budget with honest copy', () => {
    const h = setup();
    startUpdate(h);
    h.deliver({ status: 'rebooting' });
    advance(REBOOT_MS - 1);
    expect(screen.getByText('Reachy is rebooting…')).toBeTruthy();
    advance(1);
    expect(screen.getByText("Update didn't finish")).toBeTruthy();
    expect(
      screen.getByText(/did not come back online after the restart/),
    ).toBeTruthy();
  });

  it('180 s of total silence after the ack is a failure and cancels the job', () => {
    const h = setup();
    startUpdate(h);
    advance(STALL_MS - 1);
    expect(screen.getByText('Updating your Reachy…')).toBeTruthy();
    advance(1);
    expect(screen.getByText("Update didn't finish")).toBeTruthy();
    // The session is still alive on this path: the embed must be told
    // to disarm its translator and restore auto-reconnect.
    expect(h.onCancelUpdate).toHaveBeenCalledTimes(1);
  });
});

/* ─────────────────── #1316 review regressions ─────────────────── */

describe('DaemonUpdateGate review regressions (desired behavior)', () => {
  it('a progressing install must not trip the stall timeout', () => {
    // The stall timer exists to catch a daemon that acked and went
    // SILENT. An install that keeps reporting progress lines is not
    // stalled, however slow the wifi: the timer must measure time
    // since the last frame, not time since the ack.
    const h = setup();
    startUpdate(h);
    advance(STALL_MS - 10_000);
    h.deliver({ status: 'in_progress', line: 'Downloading torch (2/14)' });
    advance(STALL_MS - 10_000);
    // 340 s total, but never more than 170 s without news.
    expect(screen.getByText('Updating your Reachy…')).toBeTruthy();
    expect(screen.queryByText("Update didn't finish")).toBeNull();
    // ...and real silence after the last frame still fails honestly.
    advance(STALL_MS + 10_000);
    expect(screen.getByText("Update didn't finish")).toBeTruthy();
  });

  it('a dismissed post-timeout gate must not resurrect on a late rebooting frame', () => {
    // Repro from the review: daemon acks then goes silent, the gate
    // times out locally, the user dismisses and keeps using the app.
    // On their next NORMAL end-session the embed's still-armed
    // sessionStopped translator posts `rebooting` - which must not
    // pop a full-screen "Reachy is rebooting…" over the picker, nor
    // fire onSessionLost in the middle of a clean leave.
    const h = setup();
    startUpdate(h);
    advance(STALL_MS);
    expect(screen.getByText("Update didn't finish")).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(document.body.textContent).toBe('');

    h.deliver({ status: 'rebooting' });
    expect(screen.queryByText('Reachy is rebooting…')).toBeNull();
    expect(h.onSessionLost).not.toHaveBeenCalled();
  });

  it('dismissing the notice must not depend on a parent re-render', () => {
    // `handleDismiss` flips a ref (`clearedRef`) and sets phase to the
    // value it already holds ('idle' - the prompt is DERIVED during
    // render, phase never left idle), so React bails out and the card
    // stays on screen. It only disappears in production because the
    // shell's onDismiss happens to update parent state. The latch
    // should be renderable state, not a ref the component can't react
    // to. (Same refs-as-latches family the #1316 review flagged.)
    const h = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(h.onDismiss).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toBe('');
  });
});
