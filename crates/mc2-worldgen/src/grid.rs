//! Dense 2D scalar grids with clamped, bilinear and bicubic sampling.

#[derive(Clone, Debug, PartialEq)]
pub struct Grid2 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Grid2 {
    pub fn new(w: usize, h: usize, value: f32) -> Self {
        Self {
            w,
            h,
            data: vec![value; w * h],
        }
    }

    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> f32 + Sync) -> Self {
        use rayon::prelude::*;
        let mut data = vec![0.0; w * h];
        data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = f(x, y);
            }
        });
        Self { w, h, data }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        let i = y * self.w + x;
        self.data[i] = v;
    }

    /// Value at integer coordinates clamped to the grid.
    #[inline]
    pub fn at(&self, x: i64, y: i64) -> f32 {
        let xc = x.clamp(0, self.w as i64 - 1) as usize;
        let yc = y.clamp(0, self.h as i64 - 1) as usize;
        self.data[yc * self.w + xc]
    }

    /// Bilinear sample in cell units (cell centres at integers).
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let (xf, yf) = (x.floor(), y.floor());
        let (tx, ty) = (x - xf, y - yf);
        let (xi, yi) = (xf as i64, yf as i64);
        let a = self.at(xi, yi) + (self.at(xi + 1, yi) - self.at(xi, yi)) * tx;
        let b = self.at(xi, yi + 1) + (self.at(xi + 1, yi + 1) - self.at(xi, yi + 1)) * tx;
        a + (b - a) * ty
    }

    /// Catmull-Rom bicubic sample in cell units: C1 continuous, so amplified
    /// terrain has no creases along coarse cell boundaries.
    pub fn bicubic(&self, x: f32, y: f32) -> f32 {
        let (xf, yf) = (x.floor(), y.floor());
        let (tx, ty) = (x - xf, y - yf);
        let (xi, yi) = (xf as i64, yf as i64);
        let cr = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| {
            p1 + 0.5
                * t
                * (p2 - p0
                    + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        };
        let row = |j: i64| {
            cr(
                self.at(xi - 1, j),
                self.at(xi, j),
                self.at(xi + 1, j),
                self.at(xi + 2, j),
                tx,
            )
        };
        cr(row(yi - 1), row(yi), row(yi + 1), row(yi + 2), ty)
    }

    /// Central-difference gradient in value units per cell.
    pub fn gradient(&self, x: usize, y: usize) -> (f32, f32) {
        let (xi, yi) = (x as i64, y as i64);
        (
            (self.at(xi + 1, yi) - self.at(xi - 1, yi)) * 0.5,
            (self.at(xi, yi + 1) - self.at(xi, yi - 1)) * 0.5,
        )
    }

    pub fn min_max(&self) -> (f32, f32) {
        self.data
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samplers_interpolate_and_reproduce_nodes() {
        let g = Grid2::from_fn(8, 8, |x, y| x as f32 * 2.0 + y as f32);
        assert_eq!(g.bilinear(3.0, 4.0), 10.0);
        assert!((g.bilinear(3.5, 4.5) - 11.5).abs() < 1e-5);
        // Catmull-Rom reproduces linear functions exactly away from edges.
        assert!((g.bicubic(3.25, 4.75) - (6.5 + 4.75)).abs() < 1e-4);
        assert_eq!(g.gradient(4, 4), (2.0, 1.0));
    }
}
