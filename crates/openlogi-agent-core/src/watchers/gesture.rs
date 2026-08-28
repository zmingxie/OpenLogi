//! Background HID++ control-capture watcher, one session per online device.
//!
//! Runs [`openlogi_hid::run_capture_session`] concurrently for every device in
//! the shared capture-plan list (not just the GUI's selection), restarts a
//! session when its device's plan — route, diverted controls, thumb-wheel
//! arming — changes, and dispatches each captured input against the binding
//! maps of the device it arrived on:
//!
//! - a gesture swipe through the gesture binding map,
//! - a DPI/ModeShift or thumb-wheel-tap press through the button binding map,
//! - thumb-wheel rotation through the
//!   [`ThumbwheelScrollUp`](openlogi_core::binding::ButtonId::ThumbwheelScrollUp) /
//!   [`ThumbwheelScrollDown`](openlogi_core::binding::ButtonId::ThumbwheelScrollDown)
//!   bindings — either re-synthesised as continuous, sensitivity-scaled scroll
//!   or accumulated into a custom action,
//!
//! all via the common [`crate::runtime::ActionDispatcher`].
//!
//! Unlike the CGEventTap hook, this needs no macOS Accessibility permission —
//! the events arrive over HID++, and the bound action is synthesised the same
//! way regardless.

mod dispatch;

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use openlogi_core::config::ThumbwheelSensitivity;
use openlogi_core::scroll::ScrollDelta;
use openlogi_hid::session::gesture::{
    CaptureSessionMode, CaptureSpec, GESTURE_SOURCE_BUTTONS, TouchpadJournalStore,
};
use openlogi_hid::{
    CaptureChannel, CapturedInput, DeviceRoute, FileTouchpadJournalStore, GestureError,
    run_capture_session,
};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use self::dispatch::{InputDispatcher, WheelConfiguration};
use crate::capture_plan::{DeviceCapturePlan, SharedCapturePlans};
use crate::receiver_access::{ReceiverAccess, SessionReceiverLease};
use crate::runtime::scroll::ScrollInputHandle;
use crate::runtime::{ActionDispatcher, HidppSessionId};
use crate::touchpad_monitor::SharedTouchpadMonitor;

/// How often to re-read the active device target + thumb-wheel arming so a
/// carousel switch or a binding/sensitivity edit re-points / re-arms capture.
/// It also paces the respawn of a session that ended on its own (see `manage`).
const TARGET_POLL: Duration = Duration::from_secs(1);

/// Output capabilities shared by every HID++ gesture capture session.
#[derive(Clone)]
pub struct GestureOutputs {
    actions: ActionDispatcher,
    scroll: ScrollInputHandle,
}

impl GestureOutputs {
    /// Build gesture outputs backed by the shared action and scroll runtimes.
    #[must_use]
    pub fn new(actions: ActionDispatcher, scroll: ScrollInputHandle) -> Self {
        Self { actions, scroll }
    }

    fn cancel_session(&self, session: &HidppSessionId) {
        self.actions.cancel_hidpp_session(session);
        self.scroll.cancel_hidpp_session(session);
    }

    fn post_scroll(&self, session: &HidppSessionId, delta: ScrollDelta) {
        if !self.scroll.try_hidpp_scroll(session, delta) {
            // HID++ diversion consumed the physical input already, so direct
            // synthesis is this source's fail-open path.
            openlogi_inject::post_scroll(delta);
        }
    }
}

/// Spawn the capture-manager thread. It owns a current-thread tokio runtime that
/// keeps one capture session pointed at the active device and dispatches each
/// captured input.
pub fn spawn(
    capture_plans: SharedCapturePlans,
    capture_channel: CaptureChannel,
    receiver_access: ReceiverAccess,
    outputs: GestureOutputs,
    touchpad_monitor: SharedTouchpadMonitor,
) {
    thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                warn!(error = %e, "capture watcher: could not build tokio runtime");
                return;
            }
        };
        runtime.block_on(manage(
            capture_plans,
            capture_channel,
            receiver_access,
            outputs,
            touchpad_monitor,
        ));
    });
}

/// Whether one device's thumb wheel must be diverted over HID++ (which
/// suppresses native scroll) so we can re-synthesise its scroll or capture its
/// tap: its sensitivity leaves the default (so we scale scroll ourselves) or a
/// thumbwheel binding does.
fn thumbwheel_armed(plan: &DeviceCapturePlan) -> bool {
    plan.thumbwheel_sensitivity != ThumbwheelSensitivity::DEFAULT
        || plan.thumbwheel_bindings_nondefault
}

/// The [`CaptureSpec`] one device's session should run with right now.
fn spec_for(plan: &DeviceCapturePlan, diagnostic_touchpad: bool) -> CaptureSpec {
    let mode = match (plan.session_mode, diagnostic_touchpad) {
        (CaptureSessionMode::TouchpadRecovery, true) => CaptureSessionMode::TouchpadOnly,
        (mode, _) => mode,
    };
    CaptureSpec {
        mode,
        capture_thumbwheel: thumbwheel_armed(plan),
        // Derived from the dispatch maps, so the armed diverts and the maps
        // resolving their events can never drift apart.
        divert_gesture_sources: GESTURE_SOURCE_BUTTONS
            .into_iter()
            .filter(|(_, button)| plan.gesture_bindings.contains_key(button))
            .map(|(cid, _)| cid)
            .collect(),
        divert_buttons: plan.divert_buttons.clone(),
        capture_touchpad: plan.capture_touchpad
            || (diagnostic_touchpad && plan.touchpad_journal_id.is_some()),
        touchpad_journal_id: plan.touchpad_journal_id.clone(),
    }
}

/// Capture and wheel-state configuration that determines whether a session can
/// stay armed without leaking state across a binding epoch.
#[derive(Clone, PartialEq)]
struct SessionTarget {
    route: DeviceRoute,
    spec: CaptureSpec,
    wheel: WheelConfiguration,
    rearm_generation: u64,
}

impl SessionTarget {
    fn for_plan(plan: &DeviceCapturePlan, diagnostic_touchpad: bool) -> Self {
        Self {
            route: plan.route.clone(),
            spec: spec_for(plan, diagnostic_touchpad),
            wheel: WheelConfiguration::for_plan(plan),
            rearm_generation: plan.rearm_generation,
        }
    }
}

/// One capture session tracked by the manager.
struct RunningSession {
    id: HidppSessionId,
    target: SessionTarget,
    /// Present while the session runs; taken to request a stop. `None` means
    /// the session is draining — deliberately stopped, but its task (and the
    /// control-restore writes in its teardown) may still be in flight.
    stop: Option<oneshot::Sender<()>>,
}

struct CapturedEvent {
    session: HidppSessionId,
    input: CapturedInput,
}

struct SessionDone {
    session: HidppSessionId,
    error: Option<GestureError>,
}

#[derive(Clone)]
struct SessionChannels {
    inputs: mpsc::UnboundedSender<CapturedEvent>,
    done: mpsc::UnboundedSender<SessionDone>,
    capture: CaptureChannel,
    touchpad_journal: Option<Arc<dyn TouchpadJournalStore>>,
}

/// What a capture-session manager should do with one session-completion
/// report. Shared with the keyboard manager, which tracks its single session
/// under the same rules.
#[derive(Debug, PartialEq)]
pub(super) enum DoneAction {
    /// A stale report from a session the manager no longer tracks — ignore it.
    Ignore,
    /// The tracked session's task has fully exited: drop its entry so the next
    /// tick may arm a successor. `unexpected` is true when the exit wasn't a
    /// deliberate stop and the drop deserves a warning.
    Remove { unexpected: bool },
}

/// Decide the [`DoneAction`] for a completion report carrying `done_session`,
/// given the session the manager currently tracks for that device (if any).
///
/// Only the *current* session's report settles anything; a stale epoch belongs
/// to a session already superseded. A tracked session whose stop sender is
/// gone was stopped deliberately and is merely draining — its report frees the
/// key quietly. One still holding its stop sender exited on its own and
/// warrants a warning alongside the re-arm.
fn on_done(done_session: &HidppSessionId, live: Option<&RunningSession>) -> DoneAction {
    match live {
        Some(session) if session.id == *done_session => DoneAction::Remove {
            unexpected: session.stop.is_some(),
        },
        _ => DoneAction::Ignore,
    }
}

/// Whether an input belongs to the current, still-live session. A draining
/// session has already emitted `Cancel`, so even its correctly-tagged queued
/// events must not enter the replacement lifecycle.
fn accepts_input(input_session: &HidppSessionId, live: Option<&RunningSession>) -> bool {
    live.is_some_and(|session| session.id == *input_session && session.stop.is_some())
}

/// Whether the plan currently published for a device still describes the
/// capture session that produced an input. This closes the interval between a
/// plan publication and the manager's next teardown tick.
fn session_matches_plan(
    session: &RunningSession,
    plan: &DeviceCapturePlan,
    diagnostic_touchpad: bool,
) -> bool {
    session.target == SessionTarget::for_plan(plan, diagnostic_touchpad)
}

/// Snapshot the sessions that should be armed on this tick. Pairing owns the
/// receiver exclusively, so its request temporarily makes the wanted set
/// empty and lets the normal teardown path restore every control.
fn wanted_sessions(
    receiver_access: &ReceiverAccess,
    capture_plans: &SharedCapturePlans,
    touchpad_monitor: &SharedTouchpadMonitor,
    touchpad_journal: Option<&dyn TouchpadJournalStore>,
) -> HashMap<String, SessionTarget> {
    if receiver_access.exclusive_requested() {
        return HashMap::new();
    }
    capture_plans
        .read()
        .map(|plans| {
            plans
                .iter()
                .filter_map(|plan| {
                    let target = SessionTarget::for_plan(
                        plan,
                        touchpad_monitor.capture_requested_for(&plan.config_key),
                    );
                    if target.spec.mode == CaptureSessionMode::TouchpadRecovery {
                        let journal_id = target.spec.touchpad_journal_id.as_deref()?;
                        let journal = touchpad_journal?;
                        match journal.load(journal_id) {
                            Ok(Some(_)) => {}
                            Ok(None) => return None,
                            Err(error) => warn!(
                                key = plan.config_key,
                                error = %error,
                                "could not inspect touchpad raw-mode journal — attempting recovery"
                            ),
                        }
                    }
                    Some((plan.config_key.clone(), target))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn clear_changed_conflicts(
    blocked: &mut HashMap<String, SessionTarget>,
    wanted: &HashMap<String, SessionTarget>,
    monitor: &SharedTouchpadMonitor,
) {
    blocked.retain(|key, target| {
        let unchanged = wanted.get(key) == Some(target);
        if !unchanged {
            monitor.clear_conflict(key);
        }
        unchanged
    });
}

fn handle_session_done(
    done: &SessionDone,
    sessions: &mut HashMap<String, RunningSession>,
    blocked: &mut HashMap<String, SessionTarget>,
    monitor: &SharedTouchpadMonitor,
    dispatcher: &mut InputDispatcher,
) {
    let key = done.session.device_key();
    let DoneAction::Remove { unexpected } = on_done(&done.session, sessions.get(key)) else {
        return;
    };
    let conflict = done.error.as_ref().and_then(|error| match error {
        GestureError::TouchpadRawModeConflict { expected, actual } => Some((*expected, *actual)),
        GestureError::Hid(_)
        | GestureError::DeviceNotFound
        | GestureError::DeviceUnreachable(_)
        | GestureError::Hidpp(_) => None,
    });
    let target = sessions.get(key).map(|session| session.target.clone());
    let recovered_touchpad = done.error.is_none()
        && target
            .as_ref()
            .is_some_and(|target| target.spec.mode == CaptureSessionMode::TouchpadRecovery);
    dispatcher.cancel_session(&done.session);
    if let (Some((expected, actual)), Some(target)) = (conflict, target) {
        blocked.insert(key.to_string(), target);
        monitor.set_conflict(key, expected, actual);
        warn!(
            key,
            expected, actual, "touchpad raw-mode conflict — capture blocked until its plan changes"
        );
    } else if unexpected && !recovered_touchpad {
        warn!(key, "capture session ended unexpectedly, re-arming");
    }
    sessions.remove(key);
}

fn handle_captured_event(
    event: CapturedEvent,
    sessions: &HashMap<String, RunningSession>,
    receiver_access: &ReceiverAccess,
    capture_plans: &SharedCapturePlans,
    touchpad_monitor: &SharedTouchpadMonitor,
    input_dispatcher: &mut InputDispatcher,
) {
    let key = event.session.device_key();
    let live = sessions.get(key);
    let mut touchpad_actions_enabled = false;
    let current = accepts_input(&event.session, live)
        && !receiver_access.exclusive_requested()
        && capture_plans.read().is_ok_and(|plans| {
            plans
                .iter()
                .find(|plan| plan.config_key == key)
                .is_some_and(|plan| {
                    touchpad_actions_enabled = plan.capture_touchpad;
                    live.is_some_and(|session| {
                        session_matches_plan(
                            session,
                            plan,
                            touchpad_monitor.capture_requested_for(key),
                        )
                    })
                })
        });
    if current {
        touchpad_monitor.record(key, &event.input);
        input_dispatcher.dispatch(&event.session, event.input, touchpad_actions_enabled);
    } else {
        input_dispatcher.cancel_session(&event.session);
        debug!(
            key,
            epoch = event.session.epoch(),
            "input from a stale capture session — ignored"
        );
    }
}

/// Keep one capture session alive per online device, restarting a session when
/// its device's plan changes, and dispatch incoming inputs against the plan of
/// the device they arrived on. Runs for the lifetime of the process.
async fn manage(
    capture_plans: SharedCapturePlans,
    capture_channel: CaptureChannel,
    receiver_access: ReceiverAccess,
    outputs: GestureOutputs,
    touchpad_monitor: SharedTouchpadMonitor,
) {
    let (tx, mut rx) = mpsc::unbounded_channel::<CapturedEvent>();
    let mut sessions: HashMap<String, RunningSession> = HashMap::new();
    // A competing manager's raw-mode write blocks this exact target. The block
    // is cleared only when the plan disappears or changes, preventing a
    // write-war rearm loop against Options+.
    let mut blocked_touchpads: HashMap<String, SessionTarget> = HashMap::new();
    let mut ticker = tokio::time::interval(TARGET_POLL);
    let mut input_dispatcher = InputDispatcher::new(Arc::clone(&capture_plans), outputs);
    // Capture sessions run as detached tasks, so an unexpected exit (a transient
    // HID++ read error, a sleep-wake glitch, brief radio loss) would otherwise go
    // unnoticed. Each session reports its completion here, tagged with its device
    // key and the epoch it started under: a dead *current* session re-arms on the
    // next tick, a deliberately stopped one merely frees its key for the
    // replacement once its teardown has drained, and stale completions are
    // ignored (see `on_done`).
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<SessionDone>();
    let touchpad_journal = match FileTouchpadJournalStore::in_state_dir() {
        Ok(store) => Some(Arc::new(store) as Arc<dyn TouchpadJournalStore>),
        Err(error) => {
            warn!(error = %error, "touchpad raw-mode journal unavailable — raw capture disabled");
            None
        }
    };
    let channels = SessionChannels {
        inputs: tx,
        done: done_tx,
        capture: capture_channel,
        touchpad_journal,
    };
    // The capture-vs-pairing arbiter hands out one exclusive lease. All session
    // tasks share it through an `Arc`; the manager keeps only a `Weak` so the
    // lease frees itself when the last session exits (letting pairing proceed).
    let mut lease: std::sync::Weak<SessionReceiverLease> = std::sync::Weak::new();

    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                handle_captured_event(
                    event,
                    &sessions,
                    &receiver_access,
                    &capture_plans,
                    &touchpad_monitor,
                    &mut input_dispatcher,
                );
            }
            _ = ticker.tick() => {
                // While pairing is waiting or active, release every capture
                // session so run_pairing can own the receiver's HID node (one
                // process can't read it through two channels).
                let want = wanted_sessions(
                    &receiver_access,
                    &capture_plans,
                    &touchpad_monitor,
                    channels.touchpad_journal.as_deref(),
                );
                clear_changed_conflicts(&mut blocked_touchpads, &want, &touchpad_monitor);
                // Stop sessions whose device disappeared or whose plan changed.
                // Sending on the oneshot lets the session restore its controls.
                // A stopped session stays tracked — stop sender taken — until
                // its task reports completion below, and a tracked key is never
                // re-armed: arming the replacement while the old task may still
                // be mid-restore could interleave its divert writes with the
                // restore writes on the same device, leaving a control
                // un-diverted while the new session believes it owns it,
                // however many ticks the restore takes.
                for (key, session) in &mut sessions {
                    let keep = want.get(key).is_some_and(|target| *target == session.target);
                    if !keep && let Some(stop) = session.stop.take() {
                        input_dispatcher.cancel_session(&session.id);
                        let _ = stop.send(());
                    }
                }
                input_dispatcher.retain_devices(|key| want.contains_key(key));
                for (key, target) in want {
                    if sessions.contains_key(&key)
                        || blocked_touchpads.get(&key) == Some(&target)
                    {
                        continue;
                    }
                    // All sessions share one exclusive lease; acquire it with the
                    // first session and ride the existing one afterwards.
                    let session_lease = if let Some(existing) = lease.upgrade() {
                        existing
                    } else {
                        let Some(fresh) = receiver_access.try_acquire_for_session() else {
                            continue;
                        };
                        let fresh = Arc::new(fresh);
                        lease = Arc::downgrade(&fresh);
                        fresh
                    };
                    let id = HidppSessionId::new(&key);
                    let session = spawn_session(
                        id,
                        target,
                        session_lease,
                        &channels,
                    );
                    sessions.insert(key, session);
                }
            }
            Some(done) = done_rx.recv() => {
                // A capture session's task has fully exited — its restore writes
                // included — so dropping its entry lets the next tick start a
                // fresh session for that device; the tick fires at most once per
                // `TARGET_POLL`, which paces the respawn so a permanently failing
                // device can't hot-loop. A stale epoch (an already-superseded
                // session) is a no-op.
                handle_session_done(
                    &done,
                    &mut sessions,
                    &mut blocked_touchpads,
                    &touchpad_monitor,
                    &mut input_dispatcher,
                );
            }
        }
    }
}

/// Start one device's capture session plus its input-forwarding task, and
/// return the manager's tracking entry for it.
fn spawn_session(
    id: HidppSessionId,
    target: SessionTarget,
    lease: Arc<SessionReceiverLease>,
    channels: &SessionChannels,
) -> RunningSession {
    let (stop_tx, stop_rx) = oneshot::channel();
    // Tag this session's inputs with its device key so dispatch resolves them
    // against the right plan.
    let (session_tx, mut session_rx) = mpsc::unbounded_channel::<CapturedInput>();
    let forward = channels.inputs.clone();
    let forward_id = id.clone();
    tokio::spawn(async move {
        while let Some(input) = session_rx.recv().await {
            let _ = forward.send(CapturedEvent {
                session: forward_id.clone(),
                input,
            });
        }
    });
    let done = channels.done.clone();
    let done_id = id.clone();
    let session_route = target.route.clone();
    let session_spec = target.spec.clone();
    let slot = Arc::clone(&channels.capture);
    let touchpad_journal = channels.touchpad_journal.clone();
    tokio::spawn(async move {
        let _lease = lease;
        let backend = openlogi_hid::host::backend();
        let error = run_capture_session(
            &*backend,
            session_route,
            session_spec,
            touchpad_journal,
            session_tx,
            stop_rx,
            slot,
        )
        .await
        .err();
        if let Some(e) = error.as_ref() {
            debug!(error = %e, "capture session ended");
        }
        // Report completion so the manager can re-arm if this exit was
        // unexpected rather than a deliberate stop.
        let _ = done.send(SessionDone {
            session: done_id,
            error,
        });
    });
    RunningSession {
        id,
        target,
        stop: Some(stop_tx),
    }
}

#[cfg(test)]
mod tests;
