//! Graphics quality presets.
//!
//! Four tiers from "Realistic" to "Super Ultra Crazy Duper Realistic". Each
//! preset is a plain settings struct, so a tier is just a starting point and
//! every knob stays individually adjustable. Knobs are added here as the
//! systems they scale land; the budget table in the README is measured at
//! `UltraRealistic`.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    Realistic,
    HyperRealistic,
    UltraRealistic,
    SuperUltraCrazyDuperRealistic,
}

impl Preset {
    pub const ALL: [Preset; 4] = [
        Preset::Realistic,
        Preset::HyperRealistic,
        Preset::UltraRealistic,
        Preset::SuperUltraCrazyDuperRealistic,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Realistic => "Realistic",
            Preset::HyperRealistic => "Hyper Realistic",
            Preset::UltraRealistic => "Ultra Realistic",
            Preset::SuperUltraCrazyDuperRealistic => "Super Ultra Crazy Duper Realistic",
        }
    }

    /// Accepts the tier index (0-3) or a name ignoring case, spaces and dashes.
    pub fn parse(s: &str) -> Option<Preset> {
        if let Ok(i) = s.parse::<usize>() {
            return Self::ALL.get(i).copied();
        }
        let key: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        Self::ALL.into_iter().find(|p| {
            let name: String = p
                .name()
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .map(|c| c.to_ascii_lowercase())
                .collect();
            name == key
        })
    }

    pub fn next(self) -> Preset {
        let i = Self::ALL.iter().position(|&p| p == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    pub fn settings(self) -> Quality {
        match self {
            Preset::Realistic => Quality {
                preset: self,
                render_scale: 0.5,
                lod_pixels: 2.0,
                trace_stride: 2,
                restir_candidates: 8,
                gi: crate::indirect::GiMethod::RadianceCascades,
                cloud_steps: 32,
                cloud_light_steps: 4,
            },
            Preset::HyperRealistic => Quality {
                preset: self,
                render_scale: 0.667,
                lod_pixels: 1.5,
                trace_stride: 2,
                restir_candidates: 16,
                gi: crate::indirect::GiMethod::RadianceCascades,
                cloud_steps: 48,
                cloud_light_steps: 5,
            },
            Preset::UltraRealistic => Quality {
                preset: self,
                render_scale: 0.75,
                lod_pixels: 1.0,
                trace_stride: 1,
                restir_candidates: 32,
                gi: crate::indirect::GiMethod::RadianceCascades,
                cloud_steps: 64,
                cloud_light_steps: 6,
            },
            Preset::SuperUltraCrazyDuperRealistic => Quality {
                preset: self,
                render_scale: 1.0,
                lod_pixels: 0.5,
                trace_stride: 1,
                restir_candidates: 64,
                gi: crate::indirect::GiMethod::RadianceCascades,
                cloud_steps: 96,
                cloud_light_steps: 6,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    pub preset: Preset,
    /// Internal resolution as a fraction of the display.
    pub render_scale: f32,
    /// Primary rays stop descending once a cell covers this many pixels.
    pub lod_pixels: f32,
    /// Visibility rays trace one pixel in `trace_stride`^2 per frame once a
    /// pixel's history has settled.
    pub trace_stride: u32,
    /// Initial ReSTIR candidates per pixel (M).
    pub restir_candidates: u32,
    /// Indirect light method.
    pub gi: crate::indirect::GiMethod,
    /// Cloud ray march samples through the layer and toward the light.
    pub cloud_steps: u32,
    pub cloud_light_steps: u32,
}

impl Default for Quality {
    fn default() -> Self {
        Preset::UltraRealistic.settings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_by_index_and_name() {
        assert_eq!(Preset::parse("0"), Some(Preset::Realistic));
        assert_eq!(
            Preset::parse("super-ultra-crazy-duper-realistic"),
            Some(Preset::SuperUltraCrazyDuperRealistic)
        );
        assert_eq!(
            Preset::parse("Hyper Realistic"),
            Some(Preset::HyperRealistic)
        );
        assert_eq!(Preset::parse("4"), None);
        assert_eq!(Preset::parse("medium"), None);
    }

    #[test]
    fn tiers_scale_monotonically() {
        let s: Vec<Quality> = Preset::ALL.iter().map(|p| p.settings()).collect();
        for w in s.windows(2) {
            assert!(w[1].render_scale >= w[0].render_scale);
            assert!(w[1].lod_pixels <= w[0].lod_pixels);
            assert!(w[1].trace_stride <= w[0].trace_stride);
            assert!(w[1].restir_candidates >= w[0].restir_candidates);
            assert!(w[1].cloud_steps >= w[0].cloud_steps);
            assert!(w[1].cloud_light_steps >= w[0].cloud_light_steps);
        }
        assert_eq!(
            Preset::SuperUltraCrazyDuperRealistic.next(),
            Preset::Realistic
        );
    }
}
