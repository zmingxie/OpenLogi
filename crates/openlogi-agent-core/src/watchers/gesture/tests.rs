use super::*;
use openlogi_core::binding::{Action, Binding, ButtonId};
use openlogi_hid::session::gesture::{RawModeJournal, TouchpadJournalError, TouchpadJournalStore};

struct StubJournal(Option<RawModeJournal>);

impl TouchpadJournalStore for StubJournal {
    fn load(&self, _: &str) -> Result<Option<RawModeJournal>, TouchpadJournalError> {
        Ok(self.0)
    }

    fn save(&self, _: &str, _: RawModeJournal) -> Result<(), TouchpadJournalError> {
        Ok(())
    }

    fn clear(&self, _: &str) -> Result<(), TouchpadJournalError> {
        Ok(())
    }
}

fn route() -> DeviceRoute {
    DeviceRoute::Direct {
        vendor_id: 0x046d,
        product_id: 0xc548,
    }
}

fn session_id(epoch: u64) -> HidppSessionId {
    HidppSessionId::with_epoch("mouse-a", epoch)
}

fn stopped_session_with_epoch(epoch: u64) -> RunningSession {
    let plan = crate::capture_plan::plan_for_device(
        &openlogi_core::config::Config::default(),
        "mouse-a",
        route(),
        None,
        0,
    );
    RunningSession {
        id: session_id(epoch),
        target: SessionTarget::for_plan(&plan, false),
        stop: None,
    }
}

fn live_session_with_epoch(epoch: u64) -> RunningSession {
    let (stop, _rx) = oneshot::channel();
    RunningSession {
        stop: Some(stop),
        ..stopped_session_with_epoch(epoch)
    }
}

#[test]
fn rearms_when_the_current_session_dies() {
    assert_eq!(
        on_done(&session_id(7), Some(&live_session_with_epoch(7))),
        DoneAction::Remove { unexpected: true }
    );
}

#[test]
fn ignores_a_stale_session_superseded_by_a_restart() {
    assert_eq!(
        on_done(&session_id(6), Some(&live_session_with_epoch(7))),
        DoneAction::Ignore
    );
}

#[test]
fn ignores_a_completion_from_another_device_at_the_same_epoch() {
    assert_eq!(
        on_done(
            &HidppSessionId::with_epoch("mouse-b", 7),
            Some(&live_session_with_epoch(7))
        ),
        DoneAction::Ignore
    );
}

#[test]
fn ignores_a_completion_for_an_untracked_device() {
    assert_eq!(on_done(&session_id(7), None), DoneAction::Ignore);
}

#[test]
fn settles_a_draining_session_quietly() {
    assert_eq!(
        on_done(&session_id(7), Some(&stopped_session_with_epoch(7))),
        DoneAction::Remove { unexpected: false }
    );
}

#[test]
fn accepts_inputs_only_from_the_current_live_session() {
    assert!(accepts_input(
        &session_id(7),
        Some(&live_session_with_epoch(7))
    ));
    assert!(
        !accepts_input(&session_id(6), Some(&live_session_with_epoch(7))),
        "a superseded session's queued input is stale"
    );
    assert!(
        !accepts_input(&session_id(7), Some(&stopped_session_with_epoch(7))),
        "a draining session was already canceled"
    );
    assert!(!accepts_input(&session_id(7), None));
}

#[test]
fn rejects_input_after_the_published_capture_plan_changes() {
    let session = live_session_with_epoch(7);
    let mut plan = crate::capture_plan::plan_for_device(
        &openlogi_core::config::Config::default(),
        "mouse-a",
        session.target.route.clone(),
        None,
        0,
    );
    assert!(session_matches_plan(&session, &plan, false));

    plan.rearm_generation = 1;
    assert!(
        !session_matches_plan(&session, &plan, false),
        "an input queued before a capture-plan epoch change is stale"
    );
}

#[test]
fn wheel_configuration_changes_invalidate_the_capture_epoch() {
    let mut config = openlogi_core::config::Config::default();
    config.set_binding(
        "mouse-a",
        ButtonId::ThumbwheelScrollUp,
        Binding::Single(Action::NextTab),
    );
    let first = crate::capture_plan::plan_for_device(&config, "mouse-a", route(), None, 0);
    let mut session = live_session_with_epoch(7);
    session.target = SessionTarget::for_plan(&first, false);

    config.set_binding(
        "mouse-a",
        ButtonId::ThumbwheelScrollUp,
        Binding::Single(Action::VolumeUp),
    );
    let rebound = crate::capture_plan::plan_for_device(&config, "mouse-a", route(), None, 0);
    assert_eq!(
        spec_for(&first, false),
        spec_for(&rebound, false),
        "both custom bindings require the same HID++ diversion"
    );
    assert!(
        !session_matches_plan(&session, &rebound, false),
        "binding changes must end the epoch even when the divert set is unchanged"
    );

    session.target = SessionTarget::for_plan(&rebound, false);
    config.set_device_thumbwheel_sensitivity("mouse-a", Some(ThumbwheelSensitivity::MIN));
    let rescaled = crate::capture_plan::plan_for_device(&config, "mouse-a", route(), None, 0);
    assert_eq!(spec_for(&rebound, false), spec_for(&rescaled, false));
    assert!(
        !session_matches_plan(&session, &rescaled, false),
        "sensitivity changes must not reuse an old action threshold or cooldown"
    );
}

#[test]
fn diagnostics_temporarily_arm_only_raw_touchpad_plans() {
    let touchpad = crate::capture_plan::plan_for_device_with_touchpad(
        &openlogi_core::config::Config::default(),
        "touchpad-a",
        route(),
        None,
        Some("unit:01020304".into()),
        0,
    );
    let mouse = crate::capture_plan::plan_for_device(
        &openlogi_core::config::Config::default(),
        "mouse-a",
        route(),
        None,
        0,
    );

    assert!(!spec_for(&touchpad, false).capture_touchpad);
    assert!(spec_for(&touchpad, true).capture_touchpad);
    assert!(!spec_for(&mouse, true).capture_touchpad);
}

#[test]
fn recovery_session_is_wanted_only_while_its_journal_exists() {
    let plan = crate::capture_plan::touchpad_recovery_plan(
        "touchpad-a",
        route(),
        "unit:01020304".to_string(),
        0,
    );
    let plans = Arc::new(std::sync::RwLock::new(vec![plan]));
    let access = ReceiverAccess::default();
    let monitor = Arc::new(crate::touchpad_monitor::TouchpadMonitor::default());
    let missing = StubJournal(None);
    let existing = StubJournal(Some(RawModeJournal {
        original: 0,
        requested: 5,
        readback: Some(5),
        armed: true,
    }));

    assert!(wanted_sessions(&access, &plans, &monitor, Some(&missing)).is_empty());
    let wanted = wanted_sessions(&access, &plans, &monitor, Some(&existing));
    assert_eq!(
        wanted.get("touchpad-a").map(|target| target.spec.mode),
        Some(CaptureSessionMode::TouchpadRecovery)
    );
}

#[test]
fn managed_touchpad_session_stays_wanted_without_a_journal_record() {
    let plan = crate::capture_plan::plan_for_device_with_touchpad(
        &openlogi_core::config::Config::default(),
        "touchpad-a",
        route(),
        None,
        Some("unit:01020304".to_string()),
        0,
    );
    let plans = Arc::new(std::sync::RwLock::new(vec![plan]));
    let access = ReceiverAccess::default();
    let monitor = Arc::new(crate::touchpad_monitor::TouchpadMonitor::default());

    let wanted = wanted_sessions(&access, &plans, &monitor, None);

    assert_eq!(
        wanted.get("touchpad-a").map(|target| target.spec.mode),
        Some(CaptureSessionMode::Continuous)
    );
}

#[test]
fn diagnostic_can_temporarily_capture_a_disabled_touchpad() {
    let plan = crate::capture_plan::touchpad_recovery_plan(
        "touchpad-a",
        route(),
        "unit:01020304".to_string(),
        0,
    );
    let plans = Arc::new(std::sync::RwLock::new(vec![plan]));
    let access = ReceiverAccess::default();
    let monitor = Arc::new(crate::touchpad_monitor::TouchpadMonitor::default());
    monitor.poll("touchpad-a");

    let wanted = wanted_sessions(&access, &plans, &monitor, None);

    assert_eq!(
        wanted.get("touchpad-a").map(|target| target.spec.mode),
        Some(CaptureSessionMode::TouchpadOnly)
    );
    assert!(
        wanted
            .get("touchpad-a")
            .is_some_and(|target| target.spec.capture_touchpad)
    );
}
