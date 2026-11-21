//! Untrained family shapes. Disk figures are design labels from the v0.4 note,
//! not measurements. 40+ tok/s is a target, not a result.

#[derive(Clone, Copy, Debug)]
pub struct FamilyPreset {
    pub name: &'static str,
    pub total_params_design: u64,
    pub active_params_design: u64,
    pub disk_bytes_design: u64,
    pub ram_note: &'static str,
    pub tok_per_s_note: &'static str,
}

pub const NANO: FamilyPreset = FamilyPreset {
    name: "Cortex-Nano",
    total_params_design: 1_600_000_000,
    active_params_design: 1_600_000_000,
    disk_bytes_design: 380 * 1000 * 1000,
    ram_note: "16GB CPU long-run is comfortable for Nano",
    tok_per_s_note: "40+ tok/s is a design target, not a measurement",
};

pub const PRO: FamilyPreset = FamilyPreset {
    name: "Cortex-Pro",
    total_params_design: 7_500_000_000,
    active_params_design: 7_500_000_000,
    disk_bytes_design: 1_700_000_000,
    ram_note: "16GB is tighter for Pro",
    tok_per_s_note: "40+ tok/s is a design target, not a measurement",
};

pub const MAX: FamilyPreset = FamilyPreset {
    name: "Cortex-Max",
    total_params_design: 28_000_000_000,
    active_params_design: 4_000_000_000,
    disk_bytes_design: 6_500_000_000,
    ram_note: "16GB is tight for Max; about 4B active / about 1GB touched per token is the design",
    tok_per_s_note: "40+ tok/s is a design target, not a measurement",
};

pub fn family() -> &'static [FamilyPreset] {
    &[NANO, PRO, MAX]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_shapes_are_untrained_labels() {
        assert!(NANO.disk_bytes_design < PRO.disk_bytes_design);
        assert!(PRO.disk_bytes_design < MAX.disk_bytes_design);
        assert!(MAX.active_params_design < MAX.total_params_design);
        assert!(NANO.ram_note.contains("comfortable"));
        assert!(PRO.ram_note.contains("tighter"));
        assert!(MAX.ram_note.contains("tight"));
        for preset in family() {
            assert!(preset.tok_per_s_note.contains("target"));
            assert!(!preset.tok_per_s_note.contains("measured"));
        }
    }
}
