//! Platform-free raw-touchpad frames and gesture recognition.
//!
//! Device code validates and normalizes HID++ reports into [`TouchFrame`]s.
//! The recognizer consumes only integer micrometres and microseconds, so its
//! thresholds are independent of a touchpad's native coordinate range.

use crate::binding::ButtonId;

#[cfg(test)]
mod tests;

const TAP_MAX_DURATION_US: u64 = 250_000;
const TAP_MAX_TRAVEL_UM: u64 = 3_000;
const SWIPE_MIN_DISTANCE_UM: u64 = 10_000;
const SWIPE_MIN_SPEED_UM_PER_SECOND: u64 = 50_000;
const HORIZONTAL_SWIPE_MIN_DURATION_US: u64 = 50_000;
const VERTICAL_SWIPE_MIN_DURATION_US: u64 = 35_000;
const SWIPE_CROSS_AXIS_FLOOR_UM: u64 = 3_000;
const PINCH_MIN_SPREAD_CHANGE_UM: u64 = 8_000;
const PINCH_MIN_SPREAD_PERCENT: u64 = 8;
const MOTION_DOMINANCE_NUMERATOR: u64 = 3;
const MOTION_DOMINANCE_DENOMINATOR: u64 = 2;

/// One normalized touch contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TouchContact {
    /// Controller-assigned contact identifier.
    pub id: u8,
    /// Horizontal position from the left edge, in micrometres.
    pub x_um: u32,
    /// Vertical position from the top edge, in micrometres.
    pub y_um: u32,
}

/// One complete, normalized touchpad frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TouchFrame {
    /// Monotonic frame time, in microseconds.
    pub timestamp_us: u64,
    /// Whether the physical switch beneath the surface is pressed.
    pub button: bool,
    contacts: Box<[TouchContact]>,
}

impl TouchFrame {
    /// Build a frame, sorting contacts by ID and rejecting duplicate IDs.
    pub fn new(
        timestamp_us: u64,
        button: bool,
        mut contacts: Vec<TouchContact>,
    ) -> Result<Self, TouchFrameError> {
        contacts.sort_unstable_by_key(|contact| contact.id);
        if contacts.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err(TouchFrameError::DuplicateContactId);
        }
        Ok(Self {
            timestamp_us,
            button,
            contacts: contacts.into_boxed_slice(),
        })
    }

    /// Contacts in stable finger-ID order.
    #[must_use]
    pub fn contacts(&self) -> &[TouchContact] {
        &self.contacts
    }
}

/// Invalid normalized frame input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TouchFrameError {
    /// Two contacts in one frame carried the same controller ID.
    #[error("touchpad frame contains a duplicate contact id")]
    DuplicateContactId,
}

/// A recognized gesture in the product's 15-slot touchpad vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TouchpadGesture {
    /// Two-finger tap.
    TwoFingerTap,
    /// Two-finger pinch toward the centre.
    TwoFingerPinchIn,
    /// Two-finger pinch away from the centre.
    TwoFingerPinchOut,
    /// Three-finger tap.
    ThreeFingerTap,
    /// Three-finger upward swipe.
    ThreeFingerSwipeUp,
    /// Three-finger downward swipe.
    ThreeFingerSwipeDown,
    /// Three-finger leftward swipe.
    ThreeFingerSwipeLeft,
    /// Three-finger rightward swipe.
    ThreeFingerSwipeRight,
    /// Four-finger tap.
    FourFingerTap,
    /// Four-finger upward swipe.
    FourFingerSwipeUp,
    /// Four-finger downward swipe.
    FourFingerSwipeDown,
    /// Four-finger leftward swipe.
    FourFingerSwipeLeft,
    /// Four-finger rightward swipe.
    FourFingerSwipeRight,
    /// Four-finger pinch toward the centre.
    FourFingerPinchIn,
    /// Four-finger pinch away from the centre.
    FourFingerPinchOut,
}

impl TouchpadGesture {
    /// Binding trigger corresponding to this recognized gesture.
    #[must_use]
    pub const fn trigger(self) -> ButtonId {
        match self {
            Self::TwoFingerTap => ButtonId::TouchpadTwoFingerTap,
            Self::TwoFingerPinchIn => ButtonId::TouchpadTwoFingerPinchIn,
            Self::TwoFingerPinchOut => ButtonId::TouchpadTwoFingerPinchOut,
            Self::ThreeFingerTap => ButtonId::TouchpadThreeFingerTap,
            Self::ThreeFingerSwipeUp => ButtonId::TouchpadThreeFingerSwipeUp,
            Self::ThreeFingerSwipeDown => ButtonId::TouchpadThreeFingerSwipeDown,
            Self::ThreeFingerSwipeLeft => ButtonId::TouchpadThreeFingerSwipeLeft,
            Self::ThreeFingerSwipeRight => ButtonId::TouchpadThreeFingerSwipeRight,
            Self::FourFingerTap => ButtonId::TouchpadFourFingerTap,
            Self::FourFingerSwipeUp => ButtonId::TouchpadFourFingerSwipeUp,
            Self::FourFingerSwipeDown => ButtonId::TouchpadFourFingerSwipeDown,
            Self::FourFingerSwipeLeft => ButtonId::TouchpadFourFingerSwipeLeft,
            Self::FourFingerSwipeRight => ButtonId::TouchpadFourFingerSwipeRight,
            Self::FourFingerPinchIn => ButtonId::TouchpadFourFingerPinchIn,
            Self::FourFingerPinchOut => ButtonId::TouchpadFourFingerPinchOut,
        }
    }
}

/// Observable result of feeding one frame to [`TouchpadGestureRecognizer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureRecognition {
    /// No gesture has committed yet.
    Pending,
    /// A custom gesture committed and should fire once.
    Gesture(TouchpadGesture),
    /// Common two-finger motion dominated spread change, so firmware-native
    /// scrolling owns this stroke and OpenLogi must not fire an action.
    NativeScroll,
}

/// Pure recognizer for one touchpad stream.
#[derive(Debug, Default)]
pub struct TouchpadGestureRecognizer {
    state: StrokeState,
}

#[derive(Debug, Default)]
enum StrokeState {
    #[default]
    Idle,
    Tracking(Stroke),
    Committed,
    Cancelled,
}

#[derive(Debug)]
struct Stroke {
    finger_count: usize,
    ids: Box<[u8]>,
    starts: Box<[TouchContact]>,
    latest: Box<[TouchContact]>,
    started_at_us: u64,
    last_at_us: u64,
    start_centroid: Point,
    start_spread_um: u64,
    max_contact_travel_um: u64,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: i64,
    y: i64,
}

impl TouchpadGestureRecognizer {
    /// Feed one complete frame. A gesture is returned at most once per stroke.
    pub fn update(&mut self, frame: &TouchFrame) -> GestureRecognition {
        let count = frame.contacts.len();
        if !(2..=4).contains(&count) {
            if count == 0 {
                let _ = self.end();
            } else {
                self.state = StrokeState::Cancelled;
            }
            return GestureRecognition::Pending;
        }

        match &mut self.state {
            StrokeState::Idle => {
                self.state = StrokeState::Tracking(Stroke::new(frame));
                GestureRecognition::Pending
            }
            StrokeState::Tracking(stroke) => {
                if !stroke.accepts(frame) {
                    self.state = StrokeState::Cancelled;
                    return GestureRecognition::Pending;
                }
                stroke.advance(frame);
                let recognition = stroke.classify();
                if !matches!(recognition, GestureRecognition::Pending) {
                    self.state = match recognition {
                        GestureRecognition::Gesture(_) => StrokeState::Committed,
                        GestureRecognition::NativeScroll => StrokeState::Cancelled,
                        GestureRecognition::Pending => unreachable!(),
                    };
                }
                recognition
            }
            StrokeState::Committed | StrokeState::Cancelled => GestureRecognition::Pending,
        }
    }

    /// End the current stroke, returning a tap when it stayed short and still.
    pub fn end(&mut self) -> Option<TouchpadGesture> {
        let state = std::mem::take(&mut self.state);
        match state {
            StrokeState::Tracking(stroke) if stroke.is_tap() => stroke.tap_gesture(),
            StrokeState::Idle
            | StrokeState::Tracking(_)
            | StrokeState::Committed
            | StrokeState::Cancelled => None,
        }
    }

    /// Cancel the current stroke without producing a tap.
    pub fn cancel(&mut self) {
        self.state = StrokeState::Cancelled;
    }
}

impl Stroke {
    fn new(frame: &TouchFrame) -> Self {
        let centroid = centroid(&frame.contacts);
        Self {
            finger_count: frame.contacts.len(),
            ids: frame.contacts.iter().map(|contact| contact.id).collect(),
            starts: frame.contacts.clone(),
            latest: frame.contacts.clone(),
            started_at_us: frame.timestamp_us,
            last_at_us: frame.timestamp_us,
            start_centroid: centroid,
            start_spread_um: spread(&frame.contacts, centroid),
            max_contact_travel_um: 0,
        }
    }

    fn accepts(&self, frame: &TouchFrame) -> bool {
        frame.contacts.len() == self.finger_count
            && frame
                .contacts
                .iter()
                .map(|contact| contact.id)
                .eq(self.ids.iter().copied())
    }

    fn advance(&mut self, frame: &TouchFrame) {
        self.last_at_us = frame.timestamp_us;
        self.latest.clone_from(&frame.contacts);
        self.max_contact_travel_um = self.max_contact_travel_um.max(
            self.starts
                .iter()
                .zip(frame.contacts.iter())
                .map(|(start, current)| contact_distance(*start, *current))
                .max()
                .unwrap_or(0),
        );
    }

    fn classify(&self) -> GestureRecognition {
        let current = self.current_geometry();
        let centroid_distance = vector_length(current.dx, current.dy);
        let spread_change = current.spread_um.abs_diff(self.start_spread_um);

        if self.finger_count == 2
            && centroid_distance > TAP_MAX_TRAVEL_UM
            && dominates(centroid_distance, spread_change)
        {
            return GestureRecognition::NativeScroll;
        }

        if matches!(self.finger_count, 2 | 4)
            && spread_change >= self.pinch_threshold()
            && dominates(spread_change, centroid_distance)
        {
            return GestureRecognition::Gesture(
                self.pinch_gesture(current.spread_um >= self.start_spread_um),
            );
        }

        if matches!(self.finger_count, 3 | 4)
            && let Some(gesture) = self.swipe_gesture(current.dx, current.dy)
        {
            return GestureRecognition::Gesture(gesture);
        }

        GestureRecognition::Pending
    }

    fn current_geometry(&self) -> Geometry {
        let centroid = centroid(&self.latest);
        Geometry {
            dx: centroid.x - self.start_centroid.x,
            dy: centroid.y - self.start_centroid.y,
            spread_um: spread(&self.latest, centroid),
        }
    }

    fn pinch_threshold(&self) -> u64 {
        PINCH_MIN_SPREAD_CHANGE_UM.max(
            self.start_spread_um
                .saturating_mul(PINCH_MIN_SPREAD_PERCENT)
                / 100,
        )
    }

    fn pinch_gesture(&self, outward: bool) -> TouchpadGesture {
        match (self.finger_count, outward) {
            (2, false) => TouchpadGesture::TwoFingerPinchIn,
            (2, true) => TouchpadGesture::TwoFingerPinchOut,
            (4, false) => TouchpadGesture::FourFingerPinchIn,
            (4, true) => TouchpadGesture::FourFingerPinchOut,
            _ => unreachable!("pinches require two or four fingers"),
        }
    }

    fn swipe_gesture(&self, dx: i64, dy: i64) -> Option<TouchpadGesture> {
        let (abs_x, abs_y) = (dx.unsigned_abs(), dy.unsigned_abs());
        let (dominant, cross, min_duration) = if abs_x > abs_y {
            (abs_x, abs_y, HORIZONTAL_SWIPE_MIN_DURATION_US)
        } else {
            (abs_y, abs_x, VERTICAL_SWIPE_MIN_DURATION_US)
        };
        let duration = self.last_at_us.saturating_sub(self.started_at_us);
        let cross_limit = SWIPE_CROSS_AXIS_FLOOR_UM.max(dominant.saturating_mul(40) / 100);
        if dominant < SWIPE_MIN_DISTANCE_UM
            || cross > cross_limit
            || duration < min_duration
            || dominant.saturating_mul(1_000_000)
                < SWIPE_MIN_SPEED_UM_PER_SECOND.saturating_mul(duration)
        {
            return None;
        }
        match (self.finger_count, abs_x > abs_y, dx > 0, dy > 0) {
            (3, true, true, _) => Some(TouchpadGesture::ThreeFingerSwipeRight),
            (3, true, false, _) => Some(TouchpadGesture::ThreeFingerSwipeLeft),
            (3, false, _, true) => Some(TouchpadGesture::ThreeFingerSwipeDown),
            (3, false, _, false) => Some(TouchpadGesture::ThreeFingerSwipeUp),
            (4, true, true, _) => Some(TouchpadGesture::FourFingerSwipeRight),
            (4, true, false, _) => Some(TouchpadGesture::FourFingerSwipeLeft),
            (4, false, _, true) => Some(TouchpadGesture::FourFingerSwipeDown),
            (4, false, _, false) => Some(TouchpadGesture::FourFingerSwipeUp),
            _ => None,
        }
    }

    fn is_tap(&self) -> bool {
        self.last_at_us.saturating_sub(self.started_at_us) <= TAP_MAX_DURATION_US
            && self.max_contact_travel_um <= TAP_MAX_TRAVEL_UM
    }

    fn tap_gesture(&self) -> Option<TouchpadGesture> {
        match self.finger_count {
            2 => Some(TouchpadGesture::TwoFingerTap),
            3 => Some(TouchpadGesture::ThreeFingerTap),
            4 => Some(TouchpadGesture::FourFingerTap),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    dx: i64,
    dy: i64,
    spread_um: u64,
}

fn centroid(contacts: &[TouchContact]) -> Point {
    let count = i64::try_from(contacts.len()).unwrap_or(1);
    Point {
        x: contacts
            .iter()
            .map(|contact| i64::from(contact.x_um))
            .sum::<i64>()
            / count,
        y: contacts
            .iter()
            .map(|contact| i64::from(contact.y_um))
            .sum::<i64>()
            / count,
    }
}

fn spread(contacts: &[TouchContact], centre: Point) -> u64 {
    let count = u64::try_from(contacts.len()).unwrap_or(1);
    contacts
        .iter()
        .map(|contact| {
            vector_length(
                i64::from(contact.x_um) - centre.x,
                i64::from(contact.y_um) - centre.y,
            )
        })
        .sum::<u64>()
        / count
}

fn contact_distance(a: TouchContact, b: TouchContact) -> u64 {
    vector_length(
        i64::from(b.x_um) - i64::from(a.x_um),
        i64::from(b.y_um) - i64::from(a.y_um),
    )
}

fn vector_length(dx: i64, dy: i64) -> u64 {
    dx.unsigned_abs()
        .saturating_pow(2)
        .saturating_add(dy.unsigned_abs().saturating_pow(2))
        .isqrt()
}

fn dominates(candidate: u64, other: u64) -> bool {
    candidate.saturating_mul(MOTION_DOMINANCE_DENOMINATOR)
        > other.saturating_mul(MOTION_DOMINANCE_NUMERATOR)
}
