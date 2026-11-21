//! Day/night four-state lifecycle FSM (Phase 5).
//!
//! ```text
//! AwakeInfer --(high surprise / explicit correction)--> AwakePlastic
//! AwakePlastic --(surprise back below low / EOS)------> AwakeInfer
//! * --(idle >= 10 min)-------------------------------> IdleConsolidate
//! IdleConsolidate --(AC power + CPU idle > 80%)-------> DeepSleepTune
//! DeepSleepTune --(sleep finished)--------------------> AwakeInfer
//! * --(/sleep requested)-----------------------------> DeepSleepTune
//! ```
//!
//! The FSM is driven by explicit events (no background thread), so
//! tests are deterministic: `tick` advances the idle clock, and the
//! other events are delivered by the CLI / runtime.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsmState {
    AwakeInfer,
    AwakePlastic,
    IdleConsolidate,
    DeepSleepTune,
}

impl FsmState {
    pub fn as_str(&self) -> &'static str {
        match self {
            FsmState::AwakeInfer => "AwakeInfer",
            FsmState::AwakePlastic => "AwakePlastic",
            FsmState::IdleConsolidate => "IdleConsolidate",
            FsmState::DeepSleepTune => "DeepSleepTune",
        }
    }
}

#[derive(Clone, Debug)]
pub enum FsmEvent {
    /// A token was generated with this surprise value.
    TokenSurprise(f32),
    /// The user explicitly corrected the model / gave a new preference.
    UserCorrection,
    /// End of a sentence / turn: plasticity can settle.
    EndOfSentence,
    /// `secs` seconds passed without user activity.
    IdleTick(f64),
    /// Any user activity resets the idle clock.
    UserActivity,
    /// Power / load conditions changed.
    Power {
        ac_connected: bool,
        cpu_idle_pct: f32,
    },
    /// The user (or a schedule) requested sleep now.
    SleepRequested,
    /// A sleep consolidation run finished.
    SleepFinished,
}

pub struct RuntimeFsm {
    pub state: FsmState,
    pub surprise_low: f32,
    pub surprise_high: f32,
    idle_secs: f64,
    ac_connected: bool,
    cpu_idle_pct: f32,
}

impl RuntimeFsm {
    pub fn new(surprise_low: f32, surprise_high: f32) -> Self {
        Self {
            state: FsmState::AwakeInfer,
            surprise_low,
            surprise_high,
            idle_secs: 0.0,
            ac_connected: false,
            cpu_idle_pct: 0.0,
        }
    }

    pub fn state(&self) -> FsmState {
        self.state
    }

    pub fn idle_secs(&self) -> f64 {
        self.idle_secs
    }

    pub fn status(&self) -> String {
        format!(
            "state={} idle={:.0}s ac={} cpu_idle={:.0}%",
            self.state.as_str(),
            self.idle_secs,
            self.ac_connected,
            self.cpu_idle_pct
        )
    }

    pub fn event(&mut self, ev: FsmEvent) {
        match ev {
            FsmEvent::UserActivity => {
                self.idle_secs = 0.0;
                if self.state == FsmState::IdleConsolidate {
                    self.state = FsmState::AwakeInfer;
                }
            }
            FsmEvent::TokenSurprise(s) => {
                self.idle_secs = 0.0;
                if s >= self.surprise_high && self.state == FsmState::AwakeInfer {
                    self.state = FsmState::AwakePlastic;
                } else if s < self.surprise_low && self.state == FsmState::AwakePlastic {
                    self.state = FsmState::AwakeInfer;
                }
            }
            FsmEvent::UserCorrection => {
                self.idle_secs = 0.0;
                if self.state == FsmState::AwakeInfer {
                    self.state = FsmState::AwakePlastic;
                }
            }
            FsmEvent::EndOfSentence => {
                if self.state == FsmState::AwakePlastic {
                    self.state = FsmState::AwakeInfer;
                }
            }
            FsmEvent::IdleTick(secs) => {
                self.idle_secs += secs;
                if self.idle_secs >= 600.0
                    && matches!(self.state, FsmState::AwakeInfer | FsmState::AwakePlastic)
                {
                    self.state = FsmState::IdleConsolidate;
                }
                self.maybe_deep_sleep();
            }
            FsmEvent::Power {
                ac_connected,
                cpu_idle_pct,
            } => {
                self.ac_connected = ac_connected;
                self.cpu_idle_pct = cpu_idle_pct;
                self.maybe_deep_sleep();
            }
            FsmEvent::SleepRequested => {
                self.state = FsmState::DeepSleepTune;
            }
            FsmEvent::SleepFinished => {
                if self.state == FsmState::DeepSleepTune {
                    self.state = FsmState::AwakeInfer;
                    self.idle_secs = 0.0;
                }
            }
        }
    }

    fn maybe_deep_sleep(&mut self) {
        if self.state == FsmState::IdleConsolidate && self.ac_connected && self.cpu_idle_pct > 80.0
        {
            self.state = FsmState::DeepSleepTune;
        }
    }
}

impl Default for RuntimeFsm {
    fn default() -> Self {
        Self::new(0.02, 0.25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsm_transitions() {
        let mut fsm = RuntimeFsm::default();
        assert_eq!(fsm.state(), FsmState::AwakeInfer);
        fsm.event(FsmEvent::TokenSurprise(0.5));
        assert_eq!(fsm.state(), FsmState::AwakePlastic);
        fsm.event(FsmEvent::EndOfSentence);
        assert_eq!(fsm.state(), FsmState::AwakeInfer);
        fsm.event(FsmEvent::UserCorrection);
        assert_eq!(fsm.state(), FsmState::AwakePlastic);
        fsm.event(FsmEvent::TokenSurprise(0.001));
        assert_eq!(fsm.state(), FsmState::AwakeInfer);
        // Idle 10 minutes -> micro-sleep.
        fsm.event(FsmEvent::IdleTick(601.0));
        assert_eq!(fsm.state(), FsmState::IdleConsolidate);
        // AC + idle CPU -> deep sleep.
        fsm.event(FsmEvent::Power {
            ac_connected: true,
            cpu_idle_pct: 90.0,
        });
        assert_eq!(fsm.state(), FsmState::DeepSleepTune);
        fsm.event(FsmEvent::SleepFinished);
        assert_eq!(fsm.state(), FsmState::AwakeInfer);
        // Manual sleep request from anywhere.
        fsm.event(FsmEvent::SleepRequested);
        assert_eq!(fsm.state(), FsmState::DeepSleepTune);
        // User activity during micro-sleep wakes the model.
        let mut fsm2 = RuntimeFsm::default();
        fsm2.event(FsmEvent::IdleTick(601.0));
        assert_eq!(fsm2.state(), FsmState::IdleConsolidate);
        fsm2.event(FsmEvent::UserActivity);
        assert_eq!(fsm2.state(), FsmState::AwakeInfer);
    }
}
